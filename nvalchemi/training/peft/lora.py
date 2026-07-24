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
"""LoRA fine-tuning strategy."""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import torch
from pydantic import ConfigDict, Field, PrivateAttr, model_validator
from torch import nn

import nvalchemi.training.peft._adapter_checkpoint as adapter_checkpoint
from nvalchemi._serialization import _cls_path_of, _import_cls
from nvalchemi.hooks import Hook
from nvalchemi.training import _spec_utils as strategy_spec
from nvalchemi.training import _strategy_validation as strategy_validation
from nvalchemi.training._checkpoint import (
    _model_local_parameter_names,
    _selected_state_dict,
    _snapshot_state_dict,
)
from nvalchemi.training._spec import BaseSpec, create_model_spec_from_json
from nvalchemi.training.finetune import FineTuningStrategy, FreezeMode
from nvalchemi.training.hooks.finetune import (
    _MODULE_PATCH_HOOK_IDENTIFIER,
    ModulePatchHook,
    TrainableParameterHook,
)
from nvalchemi.training.optimizers import iter_qualified_named_parameters
from nvalchemi.training.peft import _peft
from nvalchemi.training.peft.lora_hooks import _LORA_HOOK_IDENTIFIER, LoRAApplyHook
from nvalchemi.training.peft.wrappers import (
    LoRAWrapperRegistrations,
    register_builtin_lora_wrappers,
)
from nvalchemi.training.strategy import TrainingStrategy

if TYPE_CHECKING:
    from nvalchemi.data.batch import Batch

__all__ = ["LoRAFineTuningStrategy"]

_DEFAULT_LORA_RANK = 8
_DEFAULT_LORA_ALPHA = 1.0
_DEFAULT_LORA_DROPOUT = 0.0
_DEFAULT_LORA_WRAP_MLP = False
_ADAPTER_SCHEMA_VERSION = 1
_ADAPTER_KIND = "nvalchemi_lora"
_ADAPTER_DIR = "lora"
_ADAPTER_STATE_FILENAME = "adapter.pt"
_EXTRA_TRAINABLE_STATE_NAME = "extras"


def _adapter_parameter_summary(
    adapter_state: Mapping[str, Mapping[str, torch.Tensor]],
    model: nn.Module,
) -> dict[str, int]:
    """Return parameter counts for a saved adapter state."""
    lora_state = adapter_state.get(_LORA_HOOK_IDENTIFIER, {})
    extra_state = adapter_state[_EXTRA_TRAINABLE_STATE_NAME]
    lora_parameters = sum(tensor.numel() for tensor in lora_state.values())
    extra_trainable_parameters = sum(tensor.numel() for tensor in extra_state.values())
    return {
        "lora_tensor_count": len(lora_state),
        "lora_parameter_count": lora_parameters,
        "extra_trainable_tensor_count": len(extra_state),
        "extra_trainable_parameter_count": extra_trainable_parameters,
        "trainable_parameter_count": lora_parameters + extra_trainable_parameters,
        "total_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
    }


def _remaining_lora_module_names(model: nn.Module) -> list[str]:
    """Return module names that still resolve as LoRA layers."""
    return [
        name for name, module in model.named_modules() if _peft.is_lora_layer(module)
    ]


def _lora_parameter_prefixes(models: Mapping[str, nn.Module]) -> set[str]:
    """Return qualified parameter prefixes owned by live LoRA modules."""
    prefixes: set[str] = set()
    for model_name, model in models.items():
        for module_name in _remaining_lora_module_names(model):
            prefixes.add(
                f"{model_name}." if not module_name else f"{model_name}.{module_name}."
            )
    return prefixes


def _registered_lora_parameter_names(
    registered_trainable_names: Mapping[str, frozenset[str]],
    models: Mapping[str, nn.Module],
) -> set[str]:
    """Return registered trainable parameters that belong to live LoRA modules."""
    prefixes = _lora_parameter_prefixes(models)
    if not prefixes:
        return set()
    return {
        name
        for names in registered_trainable_names.values()
        for name in names
        if any(name.startswith(prefix) for prefix in prefixes)
    }


def _parameter_names_under_target(
    parameter_names: set[str],
    target: str,
) -> set[str]:
    """Return parameter names that belong to a qualified module target."""
    prefix = f"{target}."
    return {name for name in parameter_names if name.startswith(prefix)}


def _validate_base_model_fingerprints(
    models: strategy_validation.ModelInput,
    metadata: Mapping[str, Any] | None,
) -> None:
    """Check checkpoint fingerprints against base models before LoRA is applied."""
    if not isinstance(metadata, Mapping):
        return
    saved = metadata.get("base_model_fingerprints")
    if not isinstance(saved, Mapping):
        return
    normalized = strategy_validation._normalize_models(models)
    if not isinstance(normalized, Mapping):
        return
    current = {
        name: _peft.compute_base_fingerprint(model)
        for name, model in normalized.items()
        if name in saved
    }
    mismatches = {
        name: (fingerprint, current.get(name))
        for name, fingerprint in saved.items()
        if current.get(name) != fingerprint
    }
    if mismatches:
        raise ValueError(f"LoRA checkpoint base fingerprint mismatch: {mismatches!r}.")


def _validate_restore_base_fingerprints(
    strategy: LoRAFineTuningStrategy,
    metadata: Mapping[str, Any] | None,
) -> None:
    """Check checkpoint fingerprints against LoRA metadata on a live strategy."""
    if not isinstance(metadata, Mapping):
        return
    saved = metadata.get("base_model_fingerprints")
    if not isinstance(saved, Mapping):
        return
    # Compare against the pre-adaptation fingerprints captured by LoRAApplyHook.
    # Recomputing here would fingerprint the already LoRA-wrapped strategy.
    current = adapter_checkpoint.stored_base_fingerprints(strategy)
    mismatches = {
        name: (fingerprint, current.get(name))
        for name, fingerprint in saved.items()
        if current.get(name) != fingerprint
    }
    if mismatches:
        raise ValueError(f"LoRA checkpoint base fingerprint mismatch: {mismatches!r}.")


class LoRAFineTuningStrategy(FineTuningStrategy):
    """Fine-tuning strategy that applies LoRA adapters to pretrained models.

    The strategy injects LoRA adapters before optimizer construction with
    :class:`~nvalchemi.training.peft.lora_hooks.LoRAApplyHook`, optionally applies serializable module patches with
    :class:`~nvalchemi.training.hooks.finetune.ModulePatchHook`, and marks LoRA, patched, and explicitly
    selected parameters as trainable with :class:`~nvalchemi.training.hooks.finetune.TrainableParameterHook`.
    The pretrained base parameters are frozen by default; use :attr:`trainable_patterns` to opt additional
    non-adapter parameters into training.

    Parameters
    ----------
    lora_rank : int
        LoRA adapter rank. Defaults to ``8``.
    lora_alpha : float, optional
        LoRA scaling numerator. Defaults to ``1.0``.
    lora_dropout : float, optional
        Dropout applied in the LoRA adapter path. Defaults to ``0.0``.
    lora_target_patterns : tuple[str, ...]
        Shell-style glob patterns matched against model-prefixed module paths,
        using the same ``*``, ``?``, and ``[...]`` syntax as
        ``trainable_patterns``. Dots are literal path separators. Examples:
        ``"main.model.projection"``, ``"student.model.*projection"``, or
        ``"main.model.readout*"``. Patterns without glob characters are exact
        matches.
    lora_wrap_mlp : bool, optional
        Whether to wrap the MLP layers with LoRA adapters. This is only supported for single-model strategies.
        Defaults to ``False``.
    lora_wrapper_registrations : LoRAWrapperRegistrations, optional
        Custom layer-to-wrapper registrations applied before LoRA injection.
        Defaults to ``()``.
    module_patches : dict[str, BaseSpec | torch.nn.Module], optional
        Modules to install after LoRA injection and train fully. Use
        ``BaseSpec`` values when the strategy must round-trip through
        :meth:`to_spec_dict`. Defaults to ``{}``.
    trainable_patterns : tuple[str, ...], optional
        Fully qualified parameter-name patterns for extra non-adapter parameters
        to train. Defaults to ``()``.
    freeze_mode : {"requires_grad", "optimizer_only"}
        Whether non-trainable base parameters are temporarily frozen via
        ``requires_grad=False`` or only excluded from optimizers. Defaults to
        ``"requires_grad"``.

    Raises
    ------
    ValueError
        If ``lora_target_patterns`` is not provided, or if ``freeze_patterns``
        are provided.
    """

    lora_rank: int = _DEFAULT_LORA_RANK
    lora_alpha: float = _DEFAULT_LORA_ALPHA
    lora_dropout: float = _DEFAULT_LORA_DROPOUT
    lora_target_patterns: tuple[str, ...] = Field(min_length=1)
    lora_wrap_mlp: bool = _DEFAULT_LORA_WRAP_MLP
    lora_wrapper_registrations: LoRAWrapperRegistrations = ()
    module_patches: dict[str, BaseSpec | nn.Module] = Field(default_factory=dict)
    trainable_patterns: tuple[str, ...] = ()
    freeze_mode: FreezeMode = "requires_grad"
    # The following fields are not used by the class but are present for compatibility with the base class.
    # They are excluded from serialization in Pydantic with the exclude=True flag.
    freeze_patterns: tuple[str, ...] = Field(default=(), exclude=True)
    _base_fingerprints: dict[str, str] = PrivateAttr(default_factory=dict)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_assignment=False,
        revalidate_instances="never",
    )

    @model_validator(mode="before")
    @classmethod
    def _prepend_finetuning_hooks(cls, data: Any) -> Any:
        """Prepend registration hooks for fine-tuning fields, including
        flat ``lora_*`` fields, ``module_patches``, and ``trainable_patterns``.

        ``LoRAApplyHook`` uses LoRA fields and ``lora_wrapper_registrations`` to
        generate the LoRA adapters. ``ModulePatchHook`` is used to install module patches
        after LoRA injection. ``TrainableParameterHook`` is used to mark the parameters
        that are trainable, including the additional parameters requested through
        ``trainable_patterns``. These hooks are inserted before user-supplied hooks
        so the models are prepared for fine-tuning before later hooks register.
        """
        if not isinstance(data, dict):
            return data
        normalized = dict(data)

        # Validate compatibility of user-provided fields before generating hooks.
        if normalized.get("freeze_patterns"):
            raise ValueError(
                "LoRAFineTuningStrategy does not accept freeze_patterns; LoRA "
                "freezes the pretrained base by default. Use trainable_patterns "
                "for extra non-adapter parameters."
            )
        if not normalized.get("lora_target_patterns"):
            raise ValueError("LoRAFineTuningStrategy requires lora_target_patterns.")
        # Set default values since this method is called before the user-provided fields are validated.
        normalized.setdefault("lora_rank", _DEFAULT_LORA_RANK)
        normalized.setdefault("lora_alpha", _DEFAULT_LORA_ALPHA)
        normalized.setdefault("lora_dropout", _DEFAULT_LORA_DROPOUT)
        normalized.setdefault("lora_wrap_mlp", _DEFAULT_LORA_WRAP_MLP)
        normalized.setdefault(
            "lora_wrapper_registrations",
            (),
        )

        # Generate hooks for LoRA application, module patching, and extra trainable parameters
        # to prepare for training.
        # (1) LoRA application hook
        lora_apply_hook = LoRAApplyHook(
            lora_rank=normalized["lora_rank"],
            lora_alpha=normalized["lora_alpha"],
            lora_dropout=normalized["lora_dropout"],
            lora_target_patterns=tuple(normalized.get("lora_target_patterns") or ()),
            lora_wrap_mlp=normalized["lora_wrap_mlp"],
            wrapper_registrations=tuple(normalized["lora_wrapper_registrations"]),
        )
        normalized_models = strategy_validation._normalize_models(
            normalized.get("models")
        )
        if (
            lora_apply_hook.lora_wrap_mlp
            and isinstance(normalized_models, dict)
            and len(normalized_models) > 1
        ):
            raise ValueError(
                "LoRAFineTuningStrategy does not support lora_wrap_mlp with multiple "
                "models. Use explicit lora_target_patterns to select target modules "
                "for each model instead."
            )
        generated_hooks: list[Hook] = [lora_apply_hook]

        # (2) Module patching hook
        module_patches = normalized.get("module_patches") or {}
        if module_patches:
            # Patched modules are installed after LoRA and trained fully.
            generated_hooks.append(
                ModulePatchHook(patches=module_patches, register_parameters=True)
            )

        # (3) Trainable parameter hook
        generated_hooks.append(
            TrainableParameterHook(
                trainable_patterns=tuple(normalized.get("trainable_patterns") or ()),
                freeze_mode=normalized.get("freeze_mode", "requires_grad"),
            )
        )

        # Combine generated hooks with user-provided hooks.
        normalized["hooks"] = [
            *generated_hooks,
            *list(normalized.get("hooks") or []),
        ]

        # Add freeze_patterns for compatibility with the base class.
        normalized["freeze_patterns"] = ()
        return normalized

    def to_spec_dict(self) -> dict[str, Any]:
        """Serialize this LoRA strategy into a JSON-safe training recipe.

        The resulting spec can be passed to :meth:`from_spec_dict` to rebuild
        the strategy. The spec starts with :meth:`TrainingStrategy.to_spec_dict` for
        standard training fields, then adds the user-supplied LoRA fields needed to
        recreate the strategy including generating the required setup hooks.

        Returns
        -------
        dict[str, Any]
            A JSON-safe dictionary suitable for :func:`json.dumps`.
        """
        # Bypass FineTuningStrategy.to_spec_dict because LoRA does not serialize
        # freeze_patterns. Start from the base training recipe instead
        # (e.g., optimizer_configs, num_epochs, etc.).
        spec = TrainingStrategy.to_spec_dict(self)

        # Add the user-facing LoRA fields needed to recreate the strategy.
        spec["lora_rank"] = self.lora_rank
        spec["lora_alpha"] = self.lora_alpha
        spec["lora_dropout"] = self.lora_dropout
        spec["lora_target_patterns"] = list(self.lora_target_patterns)
        spec["lora_wrap_mlp"] = self.lora_wrap_mlp
        spec["lora_wrapper_registrations"]: list[tuple[str, str]] = [
            [_cls_path_of(layer_cls), _cls_path_of(wrapper_cls)]
            for layer_cls, wrapper_cls in self.lora_wrapper_registrations
        ]

        # Add module_patches and trainable_patterns to the spec.
        if self.module_patches:
            patch_specs: dict[str, dict[str, Any]] = {}
            for target, value in self.module_patches.items():
                if not isinstance(value, BaseSpec):
                    raise TypeError(
                        "LoRAFineTuningStrategy.to_spec_dict only supports "
                        "module_patches declared as BaseSpec values; "
                        f"{target!r} is {type(value).__name__}."
                    )
                patch_specs[target] = value.model_dump()
            spec["module_patches"] = patch_specs
        spec["trainable_patterns"] = list(self.trainable_patterns)
        spec["freeze_mode"] = self.freeze_mode
        spec["base_model_fingerprints"] = adapter_checkpoint.stored_base_fingerprints(
            self
        )
        return spec

    @classmethod
    def _wrapper_registrations_from_spec(
        cls,
        strategy: dict[str, Any],
    ) -> LoRAWrapperRegistrations:
        """Restore wrapper registration class pairs from saved spec."""
        raw_registrations = strategy.get("lora_wrapper_registrations", [])
        if not isinstance(raw_registrations, list):
            raise ValueError("strategy.json lora_wrapper_registrations must be a list.")
        registrations: list[tuple[type[nn.Module], type[_peft.LoRALayer]]] = []
        for item in raw_registrations:
            if (
                not isinstance(item, list)
                or len(item) != 2
                or not all(isinstance(path, str) for path in item)
            ):
                raise ValueError(
                    "strategy.json lora_wrapper_registrations entries must be "
                    "[layer_cls_path, wrapper_cls_path] lists."
                )
            try:
                registration = (_import_cls(item[0]), _import_cls(item[1]))
            except Exception as exc:
                raise ValueError(f"Invalid wrapper registration: {exc}") from exc
            registrations.append(registration)
        return tuple(registrations)

    @classmethod
    def from_spec_dict(
        cls,
        spec: dict[str, Any],
        *,
        models: strategy_validation.ModelInput | None = None,
        hooks: list[Any] | None = None,
        training_fn: Any = None,
    ) -> LoRAFineTuningStrategy:
        """Rebuild a :class:`LoRAFineTuningStrategy` from ``to_spec_dict`` output.

        Overrides :meth:`~nvalchemi.training.finetune.FineTuningStrategy.from_spec_dict`
        to restore the LoRA-specific fields and validate the base model fingerprints.
        This override is required on the checkpoint restore path: the inherited
        :meth:`~nvalchemi.training.TrainingStrategy.from_checkpoint_dict`
        calls :meth:`from_spec_dict`, so :meth:`load_checkpoint` relies on it to
        reconstruct a LoRA strategy.

        Parameters
        ----------
        spec : dict[str, Any]
            A dict produced by :meth:`to_spec_dict`, optionally after a JSON
            round-trip.
        models : BaseModelMixin | dict[str, BaseModelMixin] | None, optional
            Runtime model override(s).
        hooks : list[Any] | None, optional
            Runtime hooks appended after generated LoRA fine-tuning hooks.
        training_fn : Any, optional
            Runtime callable or dotted-path override.

        Returns
        -------
        LoRAFineTuningStrategy
            A freshly validated LoRA fine-tuning strategy ready to :meth:`run`.
        """
        # Validate the required fields.
        required = (
            "optimizer_configs",
            "devices",
            "loss_fn_spec",
            "lora_target_patterns",
        )
        missing = [key for key in required if key not in spec]
        if missing:
            raise ValueError(
                f"from_spec_dict: spec is missing required key(s) {missing}. "
                f"Expected keys: {list(required)}."
            )

        # Reconstruct the module patches.
        module_patches = {
            target: create_model_spec_from_json(raw_spec)
            for target, raw_spec in spec.get("module_patches", {}).items()
        }

        # Reconstruct models and validate the base model fingerprints.
        model_input = strategy_spec._models_from_spec_and_overrides(
            spec.get("model_specs", {}),
            models,
            single_model_input=strategy_spec._single_model_input_from_spec(
                spec.get("single_model_input")
            ),
        )
        _validate_base_model_fingerprints(model_input, spec)

        # Reconstruct the strategy.
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
            lora_rank=int(spec.get("lora_rank", _DEFAULT_LORA_RANK)),
            lora_alpha=float(spec.get("lora_alpha", _DEFAULT_LORA_ALPHA)),
            lora_dropout=float(spec.get("lora_dropout", _DEFAULT_LORA_DROPOUT)),
            lora_target_patterns=tuple(spec.get("lora_target_patterns", ())),
            lora_wrap_mlp=bool(spec.get("lora_wrap_mlp", _DEFAULT_LORA_WRAP_MLP)),
            lora_wrapper_registrations=cls._wrapper_registrations_from_spec(spec),
            module_patches=module_patches,
            trainable_patterns=tuple(spec.get("trainable_patterns", ())),
            freeze_mode=spec.get("freeze_mode", "requires_grad"),
        )

    def to_checkpoint_dict(self) -> dict[str, Any]:
        """Serialize LoRA strategy recipe and restart counters for checkpoints."""
        self._validate_lora_adapters_present()
        return super().to_checkpoint_dict()

    def run(
        self,
        dataloader: Iterable[Batch],
    ) -> None:
        """Execute the training loop after validating LoRA adapters are present
        in case the user has merged the adapters into the base models."""
        self._validate_lora_adapters_present()
        super().run(dataloader)

    def _validate_lora_adapters_present(
        self,
        models: Mapping[str, nn.Module] | None = None,
    ) -> None:
        """Validate that registered LoRA adapter parameters still exist.

        Parameters
        ----------
        models : Mapping[str, torch.nn.Module] | None, optional
            Model mapping to inspect. Defaults to the live strategy models.
            Adapter export passes checkpoint-ready models so DDP wrappers are
            unwrapped before matching registered parameter names.
        """
        registered_trainable_names = self.get_registered_trainable_parameter_names()
        inspected_models = self.models if models is None else models
        inspected_model_names = set(inspected_models)
        if not _lora_parameter_prefixes(inspected_models):
            if _lora_parameter_prefixes(self.models):
                return
            raise RuntimeError(
                "Cannot use LoRA strategy after adapters were removed or merged; "
                "no LoRA adapter module(s) remain."
            )
        lora_parameter_names = _registered_lora_parameter_names(
            registered_trainable_names,
            inspected_models,
        )
        if not lora_parameter_names:
            raise ValueError(
                "Cannot use LoRA strategy before adapters have been applied."
            )
        inspected_lora_parameter_names = {
            name
            for name in lora_parameter_names
            if name.partition(".")[0] in inspected_model_names
        }
        if not inspected_lora_parameter_names:
            return
        current_parameter_names = {
            name
            for name, _parameter in iter_qualified_named_parameters(inspected_models)
        }
        missing_lora_parameters = sorted(
            inspected_lora_parameter_names - current_parameter_names
        )
        if missing_lora_parameters:
            raise RuntimeError(
                "Cannot use LoRA strategy after adapters were removed or merged; "
                f"missing LoRA parameter(s): {missing_lora_parameters!r}."
            )
        lora_module_prefixes = {
            f"{model_name}.{module_name}."
            for model_name, model in inspected_models.items()
            for module_name in _remaining_lora_module_names(model)
        }
        stale_lora_parameters = sorted(
            name
            for name in inspected_lora_parameter_names
            if not any(name.startswith(prefix) for prefix in lora_module_prefixes)
        )
        if stale_lora_parameters:
            raise RuntimeError(
                "Cannot use LoRA strategy after adapters were removed or merged; "
                f"stale LoRA parameter(s): {stale_lora_parameters!r}."
            )

    def save_checkpoint(
        self,
        root_folder: Path | str,
        *,
        checkpoint_index: int = -1,
        save_trainable_parameters_only: bool = True,
    ) -> int:
        """Save a restartable LoRA checkpoint.

        Parameters
        ----------
        root_folder : Path | str
            Root directory for native checkpoint manifests and component state files.
        checkpoint_index : int, optional
            Checkpoint index to write. ``-1`` auto-increments from the latest
            manifest index, or starts at ``0`` when no manifest exists.
        save_trainable_parameters_only : bool, optional
            If ``True``, save only optimizer-selected trainable model state.
            Defaults to ``True`` for smaller memory footprint. If ``False``,
            save full model state like :class:`TrainingStrategy`.

        Returns
        -------
        int
            The checkpoint index of the saved checkpoint.

        Notes
        -----
        In distributed training, this method writes immediately on the calling
        process. When saving to a shared checkpoint directory, call it from only
        one rank, typically global rank 0, and synchronize other ranks before
        loading the checkpoint. Use :class:`~nvalchemi.training.CheckpointHook`
        with ``save_trainable_parameters_only=True`` for periodic LoRA
        checkpointing with rank-zero guarding.
        """
        self._validate_lora_adapters_present()
        return super().save_checkpoint(
            root_folder,
            checkpoint_index=checkpoint_index,
            save_trainable_parameters_only=save_trainable_parameters_only,
        )

    def restore_checkpoint(
        self,
        root_folder: Path | str,
        checkpoint_index: int = -1,
        map_location: str | torch.device | None = None,
        *,
        validators: Sequence[Any] | None = None,
    ) -> Mapping[str, Any]:
        """Restore checkpoint state into this LoRA strategy instance.

        This preserves :meth:`~nvalchemi.training.TrainingStrategy.restore_checkpoint`'s
        in-place behavior while checking the LoRA base-model fingerprint before loading
        checkpoint tensors into the live model.
        """
        from nvalchemi.training._checkpoint import (
            CheckpointManifest,
            _read_strategy_metadata,
        )

        root = Path(root_folder)
        manifest = CheckpointManifest.read(root)
        resolved_index = (
            manifest.checkpoint_index if checkpoint_index == -1 else checkpoint_index
        )
        strategy_metadata = _read_strategy_metadata(
            root,
            checkpoint_index=resolved_index,
            latest_checkpoint_index=manifest.checkpoint_index,
        )
        # Validate that the saved base model fingerprints match those of the current strategy.
        _validate_restore_base_fingerprints(self, strategy_metadata)
        self._validate_lora_adapters_present()
        loaded = super().restore_checkpoint(
            root_folder,
            checkpoint_index=checkpoint_index,
            map_location=map_location,
            validators=validators,
        )
        return loaded

    @classmethod
    def _lora_config_from_spec(
        cls,
        spec: dict[str, Any],
        *,
        model_name: str | None = None,
    ) -> Any:
        """Build a local PhysicsNeMo LoRA config from serialized flat fields."""
        target_patterns = spec.get("lora_target_patterns")
        if not isinstance(target_patterns, list) or not all(
            isinstance(name, str) for name in target_patterns
        ):
            raise ValueError("LoRA strategy spec is missing lora_target_patterns.")
        if model_name is not None and any(
            target.startswith(f"{model_name}.") for target in target_patterns
        ):
            local_modules: list[str] = []
            for target in target_patterns:
                prefix, separator, module_name = target.partition(".")
                if separator and prefix == model_name:
                    local_modules.append(module_name)
            target_patterns = local_modules
        try:
            return _peft.LoRAConfig(
                rank=int(spec.get("lora_rank", _DEFAULT_LORA_RANK)),
                alpha=float(spec.get("lora_alpha", _DEFAULT_LORA_ALPHA)),
                target_modules=target_patterns,
                lora_dropout=float(spec.get("lora_dropout", _DEFAULT_LORA_DROPOUT)),
                extras_trainable=[],
                wrap_mlp=bool(spec.get("lora_wrap_mlp", _DEFAULT_LORA_WRAP_MLP)),
                init="default",
            )
        except Exception as exc:
            raise ValueError(f"Invalid LoRA fields: {exc}") from exc

    def save_adapter(
        self,
        root_folder: Path | str,
        *,
        model_name: str | None = None,
        use_ema: bool = False,
    ) -> Path:
        """Save weights of LoRA adapter(s) and extra trainable states.

        This writes a portable adapter export for applying selected model
        adaptations to a compatible base model. Use :meth:`save_checkpoint`
        instead when you need a restartable training checkpoint with strategy,
        optimizer, scheduler, hook, and runtime state.

        Parameters
        ----------
        root_folder : Path | str
            Root directory for the adapter export. Artifacts are written
            under ``lora/`` (or directly into *root_folder* when
            it already names that directory):
                {root_folder}/lora/
                  manifest.json
                  strategy.json
                  models/{model_name}/
                    adapter.pt
        model_name : str | None, optional
            The name of the specific model to save the adapter for, e.g., "main".
            Defaults to ``None`` so that all the models in the strategy are saved.
        use_ema : bool, optional
            If ``True``, export trainable state from the strategy's
            ``inference_model`` slot, which is populated by ``EMAHook``. Defaults
            to ``False`` and exports the live training model state.

        Returns
        -------
        Any
            The saved adapter directory.

        Warns
        -----
        UserWarning
            If the adapter directory already exists and is non-empty.

        Notes
        -----
        In distributed training, this method writes immediately on the calling
        process. When exporting to a shared adapter directory, call it from only
        one rank, typically global rank 0, and synchronize other ranks before
        loading the adapter.
        """
        # Create the adapter directory
        adapter_dir = adapter_checkpoint.adapter_dir(root_folder, dirname=_ADAPTER_DIR)
        adapter_checkpoint.create_dir(adapter_dir)

        if self._optimizer_parameter_names is None:
            raise ValueError(
                "LoRA trainable parameter metadata is missing. "
                "TrainableParameterHook must run before save_adapter()."
            )

        # Prepare model(s) for saving
        model_names = list(self.models) if model_name is None else [model_name]
        missing = [name for name in model_names if name not in self.models]
        if missing:
            raise KeyError(
                f"Unknown model name(s) {missing!r}; available models: "
                f"{sorted(self.models)}."
            )
        checkpoint_models = adapter_checkpoint.adapter_export_models(
            self.models,
            self.inference_model,
            model_names,
            use_ema=use_ema,
        )
        self._validate_lora_adapters_present(checkpoint_models)

        # Serialize the strategy and validate the base model fingerprints
        base_fingerprints = adapter_checkpoint.stored_base_fingerprints(self)
        strategy = self.to_spec_dict()

        # Retrieve trainable parameter names registered by source. Extra
        # trainables are optimizer-selected parameters that were not registered
        # by any source, such as parameters requested through
        # TrainableParameterHook.
        registered_trainable_names = self.get_registered_trainable_parameter_names()
        registered_parameter_names = set(
            self.get_flattened_registered_trainable_parameter_names()
        )
        registered_patch_parameter_names = set(
            registered_trainable_names.get(_MODULE_PATCH_HOOK_IDENTIFIER, ())
        )
        extra_parameter_names = set(self._optimizer_parameter_names) - (
            registered_parameter_names
        )
        patch_targets = tuple(self.module_patches)

        manifest_models: dict[str, dict[str, Any]] = {}
        for model_name in model_names:
            # Collect this model's patch targets, metadata, and trainable states.
            patch_prefix = f"{model_name}."
            named_parameters = dict(checkpoint_models[model_name].named_parameters())
            model_patch_targets = tuple(
                target
                for target in patch_targets
                if target.startswith(patch_prefix)
                and _parameter_names_under_target(
                    registered_patch_parameter_names,
                    target,
                )
            )
            adapter_state = {
                source: _snapshot_state_dict(
                    _selected_state_dict(
                        named_parameters,
                        set(_model_local_parameter_names(model_name, parameter_names)),
                    )
                )
                for source, parameter_names in registered_trainable_names.items()
            }
            adapter_state[_EXTRA_TRAINABLE_STATE_NAME] = _snapshot_state_dict(
                _selected_state_dict(
                    named_parameters,
                    set(
                        _model_local_parameter_names(model_name, extra_parameter_names)
                    ),
                )
            )
            lora_module_names = [
                name
                for name, module in checkpoint_models[model_name].named_modules()
                if _peft.is_lora_layer(module)
            ]
            model_metadata = {
                "lora_module_names": sorted(lora_module_names),
                "lora_trainable_names": sorted(
                    adapter_state.get(_LORA_HOOK_IDENTIFIER, {})
                ),
                "base_model_fingerprint": base_fingerprints[model_name],
            }

            # Create the model directory and save the trainable states and extra metadata
            model_dir = adapter_dir / "models" / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            model_entry: dict[str, Any] = {
                "metadata": model_metadata,
                "lora_parameters": sorted(adapter_state.get(_LORA_HOOK_IDENTIFIER, {})),
                "extra_trainable_parameters": sorted(
                    adapter_state[_EXTRA_TRAINABLE_STATE_NAME]
                ),
                "parameter_summary": _adapter_parameter_summary(
                    adapter_state,
                    checkpoint_models[model_name],
                ),
                "module_patches": sorted(model_patch_targets),
            }
            torch.save(adapter_state, model_dir / _ADAPTER_STATE_FILENAME)
            manifest_models[model_name] = model_entry

        # Create the manifest and strategy files
        manifest = {
            "schema_version": _ADAPTER_SCHEMA_VERSION,
            "kind": _ADAPTER_KIND,
            "models": manifest_models,
            "use_ema": use_ema,
        }
        (adapter_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        (adapter_dir / "strategy.json").write_text(json.dumps(strategy, indent=2))
        return adapter_dir

    @classmethod
    def load_adapter_into_model(
        cls,
        model: nn.Module,
        root_folder: Path | str,
        *,
        model_name: str = "main",
        merge_lora: bool = True,
        strict: bool = True,
    ) -> nn.Module:
        """Load a saved LoRA adapters and other trainable parameters into a compatible base model.

        This loads the portable adapter export produced by :meth:`save_adapter`.
        Use :meth:`load_checkpoint` or :meth:`restore_checkpoint` for native
        restartable training checkpoints.

        Parameters
        ----------
        model : torch.nn.Module
            Pristine base model to mutate in place.
        root_folder : Path | str
            Root directory for the adapter export. Artifacts are read from
            ``lora/`` or directly from *root_folder* when it already names that
            directory.
        model_name : str, optional
            Model entry to load from the saved adapters. Defaults to ``"main"``.
        merge_lora : bool, optional
            If ``True``, merge the LoRA weights into the model in place. Defaults to ``True``.
        strict : bool, optional
            If ``True``, raise on base fingerprint mismatch. When
            ``merge_lora`` is also ``True``, raise if any LoRA adapters cannot
            be merged. If ``False``, warn and continue. Defaults to ``True``.

        Returns
        -------
        torch.nn.Module
            The input model after LoRA adapters, module patches, and saved
            trainable weights have been loaded.

        Examples
        --------
        >>> from nvalchemi.training import LoRAFineTuningStrategy
        >>> finetuned_model = LoRAFineTuningStrategy.load_adapter_into_model(
        >>>     pretrained_model,
        >>>     "path/to/adapter",
        >>>     model_name="main",
        >>>     merge_lora=True
        >>> )
        """
        # Read the saved adapter manifest and strategy.
        adapter_dir = adapter_checkpoint.adapter_dir(root_folder, dirname=_ADAPTER_DIR)
        manifest, strategy = adapter_checkpoint.read_adapter_metadata(
            adapter_dir,
            adapter_kind=_ADAPTER_KIND,
            schema_version=_ADAPTER_SCHEMA_VERSION,
            target_patterns_key="lora_target_patterns",
        )
        manifest_models = manifest["models"]
        if model_name not in manifest_models:
            raise KeyError(
                f"LoRA adapter does not contain model {model_name!r}; available "
                f"adapter models: {sorted(manifest_models)}."
            )
        model_entry = manifest_models[model_name]
        if not isinstance(model_entry, dict):
            raise ValueError(f"LoRA adapter model entry {model_name!r} must be a dict.")
        metadata = model_entry.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"LoRA adapter model {model_name!r} is missing metadata.")

        # Check if the base model fingerprint matches the saved fingerprint.
        saved_fingerprint = metadata.get("base_model_fingerprint")
        if saved_fingerprint:
            current_fingerprint = _peft.compute_base_fingerprint(model)
            if current_fingerprint != saved_fingerprint:
                message = (
                    f"LoRA adapter base fingerprint mismatch for model "
                    f"{model_name!r}: saved {saved_fingerprint!r}, current "
                    f"{current_fingerprint!r}."
                )
                if strict:
                    raise ValueError(message)
                warnings.warn(message, UserWarning, stacklevel=2)

        # Register LoRA wrappers and apply LoRA adapters according to the saved strategy.
        register_builtin_lora_wrappers()
        for layer_cls, wrapper_cls in cls._wrapper_registrations_from_spec(strategy):
            _peft.register_lora_wrapper(layer_cls, wrapper_cls)
        model_strategy = dict(strategy)
        lora_modules = metadata.get("lora_module_names")
        if isinstance(lora_modules, list) and all(
            isinstance(name, str) for name in lora_modules
        ):
            model_strategy["lora_target_patterns"] = lora_modules
        _peft.apply_lora(
            model,
            cls._lora_config_from_spec(model_strategy, model_name=model_name),
        )
        adapter_state = adapter_checkpoint.load_adapter_state(
            adapter_dir, f"models/{model_name}/{_ADAPTER_STATE_FILENAME}"
        )

        # Load module patches and additional trainable parameters.
        adapter_models = {model_name: model}
        patch_targets = tuple(model_entry.get("module_patches") or ())
        if not all(isinstance(target, str) for target in patch_targets):
            raise ValueError(
                f"LoRA adapter model {model_name!r} module_patches must be strings."
            )
        patches = adapter_checkpoint.restore_module_patch_specs(strategy, patch_targets)
        if patches:
            ModulePatchHook(patches=patches).on_register(
                SimpleNamespace(models=adapter_models)
            )

        # Load LoRA, patch, and extra trainable parameters.
        combined_state: dict[str, Any] = {}
        for source in (
            _LORA_HOOK_IDENTIFIER,
            _MODULE_PATCH_HOOK_IDENTIFIER,
            _EXTRA_TRAINABLE_STATE_NAME,
        ):
            source_state = adapter_state.get(source, {})
            if source == _LORA_HOOK_IDENTIFIER and not source_state:
                raise ValueError(
                    f"LoRA adapter model {model_name!r} is missing LoRA state."
                )
            if not isinstance(source_state, dict):
                raise ValueError(
                    f"LoRA adapter model {model_name!r} {source!r} state must be a dict."
                )
            combined_state.update(source_state)
        load_result = model.load_state_dict(combined_state, strict=False)
        if load_result.unexpected_keys:
            raise RuntimeError(
                "LoRA adapter contains unexpected tensor keys for model "
                f"{model_name!r}: {sorted(load_result.unexpected_keys)}."
            )

        # Check if all the saved keys were loaded.
        missing_saved_keys = set(combined_state) & set(load_result.missing_keys)
        if missing_saved_keys:
            raise RuntimeError(
                "LoRA adapter saved tensor keys were not loaded for model "
                f"{model_name!r}: {sorted(missing_saved_keys)}."
            )

        # Merge LoRA weights into the model if requested.
        if merge_lora:
            model = cls.merge_model_inplace(model, strict=strict)

        return model

    @staticmethod
    def merge_model_inplace(model: nn.Module, *, strict: bool = True) -> nn.Module:
        """Merge LoRA weights into ``model`` in place.

        Mergeable LoRA wrappers are folded into their frozen base layers and
        removed from the model tree. The returned model is the same object
        passed in and should be treated as inference- or export-oriented rather
        than LoRA-trainable.

        Parameters
        ----------
        model : torch.nn.Module
            Model to merge LoRA weights into.
        strict : bool, optional
            If ``True``, raise on non-mergeable adapter modules. If ``False``, warn
            and continue. Defaults to ``True``.

        Returns
        -------
        torch.nn.Module
            The input model after LoRA weights have been merged.
        """
        merged = _peft.merge_lora(model)
        remaining = _remaining_lora_module_names(merged)
        if remaining:
            message = (
                "LoRA merge left non-mergeable adapter module(s) in model: "
                f"{remaining}."
            )
            if strict:
                raise RuntimeError(message)
            warnings.warn(message, UserWarning, stacklevel=2)
        return merged


LoRAFineTuningStrategy.model_rebuild()
