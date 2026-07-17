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
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from pydantic import ConfigDict, Field, model_validator
from torch import nn

import nvalchemi.training.peft._adapter_checkpoint as adapter_checkpoint
from nvalchemi._serialization import _cls_path_of, _import_cls
from nvalchemi.hooks import Hook
from nvalchemi.training import _spec_utils as strategy_spec
from nvalchemi.training import _strategy_validation as strategy_validation
from nvalchemi.training._checkpoint import (
    _create_checkpoint_snapshot,
    _snapshot_state_dict,
    _write_checkpoint_snapshot,
)
from nvalchemi.training._spec import BaseSpec, create_model_spec_from_json
from nvalchemi.training.finetune import FineTuningStrategy
from nvalchemi.training.hooks.finetune import ModulePatchHook
from nvalchemi.training.optimizers import iter_qualified_named_parameters
from nvalchemi.training.peft import _peft
from nvalchemi.training.peft._peft import LoRAConfig
from nvalchemi.training.peft.hooks import (
    LoRAApplyHook,
    LoRATrainableParameterHook,
    _lora_apply_metadata,
)
from nvalchemi.training.peft.wrappers import (
    LoRAWrapperRegistration,
    LoRAWrapperRegistrations,
    register_builtin_lora_wrappers,
)
from nvalchemi.training.strategy import TrainingStrategy

__all__ = ["LoRAConfig", "LoRAFineTuningStrategy"]


def _validate_no_extras_trainable(lora_config: Any) -> None:
    """Reject PhysicsNeMo LoRA extras unsupported by ``LoRAFineTuningStrategy``."""
    extras_trainable = (
        lora_config.get("extras_trainable", [])
        if isinstance(lora_config, dict)
        else getattr(lora_config, "extras_trainable", [])
    )
    if extras_trainable:
        raise ValueError(
            "LoRAFineTuningStrategy does not support LoRAConfig's extras_trainable. "
            "Use trainable_patterns or module_patches to select extra trainable parameters."
        )


def _shared_lora_config_from_metadata(
    lora_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Reconstruct a shared LoRA config spec from saved adapter metadata for all models.

    Parameters
    ----------
    lora_metadata : dict[str, Any]
        Mapping from model name to adapter metadata, populated as
        ``workflow._lora_metadata`` by ``LoRAApplyHook.on_register``
        through ``_lora_apply_metadata``.

    Returns
    -------
    dict[str, Any]
        A shared LoRA config dict suitable for reconstructing ``LoRAConfig``.

    """
    if not lora_metadata:
        raise ValueError("LoRA adapter metadata is missing.")

    shared: dict[str, Any] | None = None
    shared_model_name: str | None = None
    shared_metadata: dict[str, Any] | None = None
    shared_lora_modules: list[str] = []
    for model_name, metadata in lora_metadata.items():
        if not isinstance(metadata, dict):
            raise TypeError(
                "LoRA adapter metadata entries must be dictionaries; "
                f"{model_name!r} is {type(metadata).__name__}."
            )
        _validate_no_extras_trainable(metadata)
        lora_modules = metadata.get("lora_modules")
        if not isinstance(lora_modules, list) or not all(
            isinstance(name, str) for name in lora_modules
        ):
            raise ValueError(
                f"LoRA adapter model {model_name!r} metadata must contain lora_modules."
            )
        for module_name in lora_modules:
            if module_name not in shared_lora_modules:
                shared_lora_modules.append(module_name)

        comparable = {
            "rank": metadata.get("rank"),
            "alpha": metadata.get("alpha"),
            "lora_dropout": metadata.get("lora_dropout", 0.0),
            "wrap_mlp": metadata.get("wrap_mlp", False),
        }
        if shared is None:
            shared = comparable
            shared_model_name = model_name
            shared_metadata = dict(metadata)
        elif comparable != shared:
            raise ValueError(
                "LoRAFineTuningStrategy.to_spec_dict requires all models to "
                "share one resolved LoRA config. Models "
                f"{shared_model_name!r} and {model_name!r} resolved different "
                "LoRA metadata."
            )

    if shared_metadata is None:
        raise ValueError("LoRA adapter metadata is missing.")
    return {
        "rank": shared_metadata.get("rank"),
        "alpha": shared_metadata.get("alpha"),
        "target_modules": shared_lora_modules,
        "lora_dropout": shared_metadata.get("lora_dropout", 0.0),
        "extras_trainable": [],
        "wrap_mlp": shared_metadata.get("wrap_mlp", False),
        "init": shared_metadata.get("init", "default"),
    }


def _remaining_lora_module_names(model: nn.Module) -> list[str]:
    """Return module names that still resolve as LoRA layers."""
    return [
        name for name, module in model.named_modules() if _peft.is_lora_layer(module)
    ]


def _validate_loaded_base_fingerprints(
    strategy: LoRAFineTuningStrategy,
    metadata: Mapping[str, Any] | None,
) -> None:
    """Check saved LoRA base fingerprints against the reconstructed strategy."""
    if not isinstance(metadata, Mapping):
        return
    saved = metadata.get("base_model_fingerprints")
    if not isinstance(saved, Mapping):
        return
    current = adapter_checkpoint.base_fingerprints(strategy)
    mismatches = {
        name: (fingerprint, current.get(name))
        for name, fingerprint in saved.items()
        if current.get(name) != fingerprint
    }
    if mismatches:
        raise ValueError(f"LoRA checkpoint base fingerprint mismatch: {mismatches!r}.")


class LoRAFineTuningStrategy(FineTuningStrategy):
    """Fine-tuning strategy that applies LoRA adapters to pretrained models.

    The strategy injects LoRA adapters before optimizer construction, optionally
    applies serializable module patches, and marks LoRA, patched, and explicitly
    selected parameters as trainable. The pretrained base parameters are frozen
    by default; use ``trainable_patterns`` to opt additional non-adapter
    parameters into training.

    Parameters
    ----------
    lora_config : LoRAConfig
        LoRA configuration used to wrap target modules in each model.
    wrapper_registrations : LoRAWrapperRegistrations, optional
        Custom layer-to-wrapper registrations applied before LoRA injection.
    module_patches : dict[str, BaseSpec | torch.nn.Module], optional
        Modules to install after LoRA injection and train fully. Use
        ``BaseSpec`` values when the strategy must round-trip through
        :meth:`to_spec_dict`.
    trainable_patterns : tuple[str, ...], optional
        Fully qualified parameter-name patterns for extra non-adapter parameters
        to train.

    Raises
    ------
    ValueError
        If ``lora_config`` is missing, if ``freeze_patterns`` or unsupported
        ``freeze_mode`` values are provided, or if ``lora_config`` uses
        ``extras_trainable``.
    """

    lora_config: LoRAConfig
    wrapper_registrations: LoRAWrapperRegistrations = ()
    module_patches: dict[str, BaseSpec | nn.Module] = Field(default_factory=dict)
    trainable_patterns: tuple[str, ...] = ()
    # The following fields are not used by the class but are present for compatibility with the base class.
    # They are excluded from serialization in Pydantic with the exclude=True flag.
    freeze_patterns: tuple[str, ...] = Field(default=(), exclude=True)
    freeze_mode: str = Field(default="requires_grad", exclude=True)

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
        ``lora_config``, ``module_patches``, and ``trainable_patterns``.

        ``lora_config`` and ``wrapper_registrations`` produce ``LoRAApplyHook``;
        ``module_patches`` produces an optional ``ModulePatchHook``; and
        ``trainable_patterns`` produces ``LoRATrainableParameterHook``. Generated
        hooks are inserted before user-supplied hooks so the models are prepared for
        fine-tuning before later hooks register.
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
        if "freeze_mode" in normalized and normalized["freeze_mode"] != "requires_grad":
            raise ValueError(
                "LoRAFineTuningStrategy does not accept freeze_mode; LoRA "
                "trainability is controlled by adapter parameters, module_patches, "
                "and trainable_patterns."
            )
        if normalized.get("lora_config") is None:
            raise ValueError("LoRAFineTuningStrategy requires lora_config.")
        _validate_no_extras_trainable(normalized["lora_config"])

        module_patches = normalized.get("module_patches") or {}

        # Generate hooks for LoRA application, module patching, and extra trainable parameters
        # to prepare for training.
        generated_hooks: list[Hook] = [
            LoRAApplyHook(
                lora_config=normalized.get("lora_config"),
                wrapper_registrations=tuple(
                    normalized.get("wrapper_registrations") or ()
                ),
            )
        ]
        if module_patches:
            # Patched modules are installed after LoRA and trained fully.
            generated_hooks.append(ModulePatchHook(patches=module_patches))
        generated_hooks.append(
            LoRATrainableParameterHook(
                trainable_patterns=tuple(normalized.get("trainable_patterns") or ()),
                patched_module_paths=tuple(module_patches),
            )
        )

        # Combine generated hooks with user-provided hooks.
        normalized["hooks"] = [
            *generated_hooks,
            *list(normalized.get("hooks") or []),
        ]

        # Add freeze_patterns and freeze_mode for compatibility with the base class.
        normalized["freeze_patterns"] = ()
        normalized["freeze_mode"] = "requires_grad"
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
        # freeze_patterns and freeze_mode. Start from the base training recipe instead
        # (e.g., optimizer_configs, num_epochs, etc.).
        spec = TrainingStrategy.to_spec_dict(self)

        # Add the user-supplied LoRA fields needed to recreate the strategy.
        # Step 1: serialize LoRA application inputs.
        # Save a constructor-shaped config with resolved target modules instead
        # of the original lora_config, which may contain non-serializable
        # selectors.
        metadata = dict(getattr(self, "_lora_metadata", {}))
        spec["lora_config"]: dict[str, Any] = _shared_lora_config_from_metadata(
            metadata
        )
        spec["wrapper_registrations"]: list[tuple[str, str]] = [
            [_cls_path_of(layer_cls), _cls_path_of(wrapper_cls)]
            for layer_cls, wrapper_cls in self.wrapper_registrations
        ]

        # Step 2: module_patches and trainable_patterns
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
        return spec

    @classmethod
    def _lora_config_from_spec(cls, spec: dict[str, Any]) -> Any:
        """Build LoRA config from a serialized spec."""
        lora_config = spec.get("lora_config")
        if not isinstance(lora_config, dict):
            raise ValueError("LoRA strategy spec is missing lora_config.")
        try:
            return LoRAConfig(**lora_config)
        except Exception as exc:
            raise ValueError(f"Invalid LoRA config: {exc}") from exc

    @classmethod
    def _wrapper_registrations_from_spec(
        cls,
        strategy: dict[str, Any],
    ) -> LoRAWrapperRegistrations:
        """Restore wrapper registration class pairs from saved spec."""
        raw_registrations = strategy.get("wrapper_registrations", [])
        if not isinstance(raw_registrations, list):
            raise ValueError("strategy.json wrapper_registrations must be a list.")
        registrations: list[LoRAWrapperRegistration] = []
        for item in raw_registrations:
            if (
                not isinstance(item, list)
                or len(item) != 2
                or not all(isinstance(path, str) for path in item)
            ):
                raise ValueError(
                    "strategy.json wrapper_registrations entries must be "
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
        required = ("optimizer_configs", "devices", "loss_fn_spec", "lora_config")
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
            lora_config=cls._lora_config_from_spec(spec),
            wrapper_registrations=cls._wrapper_registrations_from_spec(spec),
            module_patches=module_patches,
            trainable_patterns=tuple(spec.get("trainable_patterns", ())),
        )

    def to_checkpoint_dict(self) -> dict[str, Any]:
        """Serialize LoRA strategy recipe and restart counters for checkpoints."""
        self._validate_checkpointable_lora_state()
        metadata = super().to_checkpoint_dict()
        metadata["base_model_fingerprints"] = adapter_checkpoint.base_fingerprints(self)
        return metadata

    def _validate_checkpointable_lora_state(self) -> None:
        """Validate that the live model tree still matches LoRA metadata."""
        # TODO: Check this.
        metadata = getattr(self, "_lora_metadata", None)
        if not isinstance(metadata, dict) or not metadata:
            raise ValueError(
                "Cannot checkpoint LoRA strategy before adapters have been applied."
            )
        missing_lora = [
            model_name
            for model_name in metadata
            if model_name in self.models
            and not _remaining_lora_module_names(self.models[model_name])
        ]
        if missing_lora:
            raise RuntimeError(
                "Cannot checkpoint LoRA strategy after adapters were merged into "
                f"model(s) {missing_lora!r}; save a plain exported model instead."
            )

    def save_checkpoint(
        self,
        root_folder: Path | str,
        *,
        checkpoint_index: int = -1,
        include_base_parameters: bool = False,
    ) -> int:
        """Save a restartable LoRA checkpoint.

        Parameters
        ----------
        root_folder : Path | str
            Root directory for native checkpoint manifests and component state files.
        checkpoint_index : int, optional
            Checkpoint index to write. ``-1`` auto-increments from the latest
            manifest index, or starts at ``0`` when no manifest exists.
        include_base_parameters : bool, optional
            If ``True``, save full model state like :class:`TrainingStrategy`.
            Defaults to ``False`` and saves only LoRA adapters, module patches,
            and extra trainable parameters for smaller memory footprint.

        Returns
        -------
        int
            The checkpoint index of the saved checkpoint.
        """
        if include_base_parameters:
            return super().save_checkpoint(
                root_folder,
                checkpoint_index=checkpoint_index,
            )
        snapshot = _create_checkpoint_snapshot(
            root_folder,
            checkpoint_index=checkpoint_index,
            strategy=self,
        )
        adapter_checkpoint.filter_lora_snapshot(snapshot, self)
        return _write_checkpoint_snapshot(root_folder, snapshot)

    @classmethod
    def load_checkpoint(
        cls,
        root_folder: Path | str,
        checkpoint_index: int = -1,
        map_location: str | torch.device | None = None,
        **kwargs: Any,
    ) -> LoRAFineTuningStrategy:
        """Load a restartable LoRA fine-tuning checkpoint.
        
        Parameters
        ----------
        root_folder : Path | str
            Root directory for native checkpoint manifests and component state files.
        checkpoint_index : int, optional
            Checkpoint index to load. ``-1`` auto-increments from the latest
            manifest index, or starts at ``0`` when no manifest exists.
        map_location : str | torch.device | None, optional
            Map location for the checkpoint.
        **kwargs : Any, optional
            Additional keyword arguments for :func:`nvalchemi.training._checkpoint.load_checkpoint`.
        
        Returns
        -------
        LoRAFineTuningStrategy
            A freshly validated LoRA fine-tuning strategy ready to :meth:`run`.
        """
        from nvalchemi.training._checkpoint import load_checkpoint

        loaded = load_checkpoint(
            root_folder,
            checkpoint_index=checkpoint_index,
            map_location=map_location,
            **kwargs,
        )
        strategy = loaded.get("strategy")
        if not isinstance(strategy, cls):
            raise TypeError(
                f"Loaded strategy has type {type(strategy).__name__}, expected "
                f"{cls.__name__}."
            )
        _validate_loaded_base_fingerprints(
            strategy,
            loaded.get("strategy_metadata"),
        )
        return strategy

    def save_adapter(
        self,
        root_folder: Path | str,
        *,
        model_name: str | None = None,
        use_ema: bool = False,
    ) -> Path:
        """Save weights of LoRA adapter(s) and extra trainable states.

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
                    lora.pt
                    extras.pt
                    patches.pt
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
        """
        # Create the adapter directory
        adapter_dir = adapter_checkpoint.adapter_dir(root_folder)
        adapter_checkpoint.create_dir(adapter_dir)

        # Check saved metadata
        metadata = getattr(self, "_lora_metadata", None)
        if not isinstance(metadata, dict) or not metadata:
            raise ValueError(
                "LoRA adapter metadata is missing. Construct "
                "LoRAFineTuningStrategy with LoRAApplyHook before save_adapter()."
            )
        if self._optimizer_parameter_names is None:
            raise ValueError(
                "LoRA trainable parameter metadata is missing. "
                "LoRATrainableParameterHook must run before save_adapter()."
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

        # Serialize the strategy and track the base model fingerprints
        base_model_fingerprints: dict[str, str] = {}
        for model_name in model_names:
            model_metadata = metadata.get(model_name)
            if not isinstance(model_metadata, dict):
                raise ValueError(
                    f"LoRA adapter metadata is missing for model {model_name!r}."
                )
            base_model_fingerprint = model_metadata.get("base_model_fingerprint")
            if (
                not isinstance(base_model_fingerprint, str)
                or not base_model_fingerprint
            ):
                raise ValueError(
                    "LoRA adapter metadata is missing base_model_fingerprint for "
                    f"model {model_name!r}."
                )
            base_model_fingerprints[model_name] = base_model_fingerprint
        strategy = self.to_spec_dict()
        strategy["base_model_fingerprints"] = base_model_fingerprints

        # Retrieve trainable parameters for export
        named_parameters = dict(iter_qualified_named_parameters(checkpoint_models))
        lora_parameter_names = set(
            getattr(self, "_lora_adapter_parameter_names", set())
        )
        extra_parameter_names = set(
            getattr(self, "_extra_trainable_parameter_names", set())
        )
        patch_targets = tuple(self.module_patches)

        manifest_models: dict[str, dict[str, Any]] = {}
        for model_name in model_names:
            # Collect this model's patch targets, metadata, and trainable states.
            model_patch_targets = adapter_checkpoint.patch_targets_for_model(
                model_name,
                patch_targets,
            )
            model_metadata = metadata.get(model_name, {})
            if not isinstance(model_metadata, dict):
                model_metadata = {}
            lora_state = adapter_checkpoint.model_relative_parameter_state(
                model_name,
                lora_parameter_names,
                named_parameters,
            )
            extra_state = adapter_checkpoint.model_relative_parameter_state(
                model_name,
                extra_parameter_names,
                named_parameters,
            )

            # Create the model directory and save the trainable states and extra metadata
            model_dir = adapter_dir / "models" / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            model_entry: dict[str, Any] = {
                "metadata": model_metadata,
                "lora_parameters": sorted(lora_state),
                "extra_trainable_parameters": sorted(extra_state),
                "module_patches": sorted(model_patch_targets),
            }
            if lora_state:
                torch.save(_snapshot_state_dict(lora_state), model_dir / "lora.pt")
                model_entry["lora_state"] = f"models/{model_name}/lora.pt"
            if extra_state:
                torch.save(_snapshot_state_dict(extra_state), model_dir / "extras.pt")
                model_entry["extras_state"] = f"models/{model_name}/extras.pt"
            patches_state = {
                target: _snapshot_state_dict(
                    adapter_checkpoint.resolve_child_module(
                        checkpoint_models, target
                    ).state_dict()
                )
                for target in model_patch_targets
            }
            if patches_state:
                torch.save(patches_state, model_dir / "patches.pt")
                model_entry["patches_state"] = f"models/{model_name}/patches.pt"
            manifest_models[model_name] = model_entry

        # Create the manifest and strategy files
        manifest = {
            "schema_version": adapter_checkpoint.ADAPTER_SCHEMA_VERSION,
            "kind": adapter_checkpoint.ADAPTER_KIND,
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
        strict : bool, optional
            If ``True``, raise on base fingerprint mismatch. If ``False``, warn
            and continue. Defaults to ``True``.

        Returns
        -------
        torch.nn.Module
            The input model after LoRA adapters, module patches, and saved
            trainable weights have been loaded.
        """
        # Read the saved adapter manifest and strategy.
        adapter_dir = adapter_checkpoint.adapter_dir(root_folder)
        manifest, strategy = adapter_checkpoint.read_adapter_metadata(adapter_dir)
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
        _peft.apply_lora(model, cls._lora_config_from_spec(strategy))

        # Load module patches and trainable parameters.
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

        patches_state: dict[str, Any] = {}
        patches_state_path = model_entry.get("patches_state")
        if patch_targets:
            if not isinstance(patches_state_path, str):
                raise ValueError(
                    f"LoRA adapter model {model_name!r} must reference "
                    "patches_state when module_patches are present."
                )
            patches_state = adapter_checkpoint.load_adapter_state(
                adapter_dir, patches_state_path
            )
            for target in patch_targets:
                if target not in patches_state:
                    raise ValueError(
                        f"LoRA adapter patches_state is missing target {target!r}."
                    )
                patch_module = adapter_checkpoint.resolve_child_module(
                    adapter_models, target
                )
                patch_module.load_state_dict(patches_state[target], strict=True)

        # Load LoRA weights and extra trainable parameters.
        lora_state_path = model_entry.get("lora_state")
        if not isinstance(lora_state_path, str):
            raise ValueError(
                f"LoRA adapter model {model_name!r} is missing lora_state."
            )
        lora_state = adapter_checkpoint.load_adapter_state(adapter_dir, lora_state_path)
        extras_state_path = model_entry.get("extras_state")
        extras_state = (
            adapter_checkpoint.load_adapter_state(adapter_dir, extras_state_path)
            if isinstance(extras_state_path, str)
            else {}
        )
        combined_state = {**lora_state, **extras_state}
        load_result = model.load_state_dict(combined_state, strict=False)
        if load_result.unexpected_keys:
            raise RuntimeError(
                "LoRA adapter contains unexpected tensor keys for model "
                f"{model_name!r}: {sorted(load_result.unexpected_keys)}."
            )

        # Check if all the saved keys were loaded.
        saved_keys = set(combined_state)
        missing_saved_keys = saved_keys & set(load_result.missing_keys)
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
