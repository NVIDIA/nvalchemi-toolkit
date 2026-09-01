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
from typing import Annotated, Any

import torch
from pydantic import Field, PrivateAttr, model_validator

from nvalchemi.training import _spec_utils as strategy_spec
from nvalchemi.training import _strategy_validation as strategy_validation
from nvalchemi.training._spec import BaseSpec, create_model_spec_from_json
from nvalchemi.training.hooks.finetune import (
    FineTuningSummaryHook,
    FreezeMode,
    ModulePatchHook,
    TrainableParameterHook,
)
from nvalchemi.training.peft.config import (
    PeftConfig,
    build_peft_setup_hooks,
)
from nvalchemi.training.peft.fingerprints import (
    BaseFingerprintHook,
    validate_base_fingerprints,
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

    * ``peft_config`` becomes PEFT-specific registration hooks, for example
      LoRA adapter injection.
    * ``module_patches`` becomes a :class:`ModulePatchHook`.
    * ``freeze_patterns`` / ``trainable_patterns`` become a
      :class:`TrainableParameterHook`.

    PEFT methods are configured with ``peft_config``; for example,
    :class:`nvalchemi.training.peft.lora.LoRAConfig` injects LoRA adapters into
    matching linear modules. When ``peft_config`` is provided, ``freeze_patterns``
    must be empty since the base model is considered frozen by default and PEFT
    hooks register adapter parameters as trainable. Use ``trainable_patterns``
    to select any additional existing model parameters that should remain
    trainable.

    Module patch targets are fully-qualified paths of the form
    ``"<model_key>.<module_path>.<child>"``, for example
    ``"main.model.readouts.1.linear"``. The parent path must already exist.
    The final child is replaced when it is an existing ``torch.nn.Module`` or
    added when missing. Use :func:`nvalchemi.training.create_model_spec` for
    module patches that must round-trip through :meth:`to_spec_dict`; direct
    ``torch.nn.Module`` instances are supported at runtime but are rejected by
    serialization. Module patches register their parameters as trainable by
    default, so pattern filters do not need to include them and
    ``freeze_patterns`` do not exclude them.

    Parameter patterns are matched against fully-qualified names such as
    ``"main.model.readouts.1.linear.weight"``. ``trainable_patterns`` alone is
    an allow-list: only matching parameters remain trainable and enter
    optimizers. Module patch parameters and PEFT adapter parameters are registered
    as trainable separately, as described above. When ``freeze_patterns`` is also
    supplied, matching parameters are excluded first, then ``trainable_patterns`` are
    re-included. With the default ``freeze_mode="requires_grad"``, excluded parameters
    are temporarily marked ``requires_grad=False`` during :meth:`run` and restored
    afterward. Use ``freeze_mode="optimizer_only"`` when excluded parameters
    should still receive gradients but must not be updated by optimizers.

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

    peft_config: PeftConfig | None = Field(
        default=None,
        description=(
            "Parameter-efficient fine-tuning configuration. When provided, its setup "
            "hooks run before module patches and trainable parameter selection."
        ),
    )
    module_patches: dict[str, BaseSpec | torch.nn.Module] = Field(
        default_factory=dict,
        description="Ordered module patches applied before optimizer construction.",
    )
    freeze_patterns: Annotated[
        tuple[str, ...],
        Field(
            description=(
                "Glob patterns excluded from training. Exclusions can be "
                "re-included by ``trainable_patterns``. Must be empty when "
                "``peft_config`` is provided."
            )
        ),
    ] = ()
    trainable_patterns: Annotated[
        tuple[str, ...],
        Field(
            description=(
                "Glob patterns for model parameters to keep trainable. Without "
                "``freeze_patterns``, matching parameters form the allow-list, "
                "together with parameters registered as trainable by setup hooks. "
                "Module patches and, when ``peft_config`` is configured, PEFT "
                "adapters register their parameters automatically; use these "
                "patterns to select any additional existing model parameters."
            )
        ),
    ] = ()
    freeze_mode: Annotated[
        FreezeMode,
        Field(
            description=(
                "Whether excluded parameters are temporarily frozen via "
                "``requires_grad=False`` or only excluded from optimizers. "
                'Defaults to ``"requires_grad"``.'
            )
        ),
    ] = "requires_grad"
    compute_base_fingerprints: bool = Field(
        default=True,
        description=(
            "Used only when peft_config is provided. If True, compute base-model "
            "fingerprints and include them in strategy metadata for compatibility "
            "checks when loading PEFT checkpoints into a fresh base model."
        ),
    )

    # Run-time states populated by hooks (e.g., ModulePatchHook and TrainableParameterHook) to keep track
    # of trainable and managed parameter names so that the individual hook implementations can be kept modular.
    # PrivateAttr is used to avoid serializing these states to the checkpoint.
    _registered_trainable_parameter_names: dict[str, frozenset[str]] = PrivateAttr(
        default_factory=dict
    )
    _registered_managed_parameter_names: dict[str, frozenset[str]] = PrivateAttr(
        default_factory=dict
    )
    _base_fingerprints: dict[str, str] = PrivateAttr(default_factory=dict)
    _peft_details: dict[str, Any] = PrivateAttr(default_factory=dict)
    _trainable_parameter_summary: dict[str, dict[str, int]] = PrivateAttr(
        default_factory=dict
    )

    @property
    def trainable_parameter_summary(self) -> Mapping[str, dict[str, int]]:
        """Return trainable parameter counts grouped by registration source."""
        return self._trainable_parameter_summary

    @model_validator(mode="before")
    @classmethod
    def _prepend_finetuning_hooks(cls, data: Any) -> Any:
        """Convert convenience fields into registration-time hooks."""
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        generated: list[Any] = []

        # (1) PEFT-related hooks are prepended first if peft_config is provided.
        peft_config = normalized.get("peft_config")
        if isinstance(peft_config, Mapping):
            peft_config = PeftConfig.from_spec_dict(peft_config)
            normalized["peft_config"] = peft_config
        if peft_config is not None:
            if not isinstance(peft_config, PeftConfig):
                raise TypeError(
                    "FineTuningStrategy peft_config must be a PeftConfig; "
                    f"got {type(peft_config).__name__}."
                )
            if tuple(normalized.get("freeze_patterns") or ()):
                raise ValueError(
                    "FineTuningStrategy with peft_config does not accept "
                    "freeze_patterns; PEFT freezes the base model by default."
                )
            if normalized.get("compute_base_fingerprints", True):
                generated.append(BaseFingerprintHook())
            generated.extend(build_peft_setup_hooks(peft_config, normalized))

        # (2) Module patching hook is prepended next.
        module_patches = normalized.get("module_patches") or {}
        if module_patches:
            generated.append(ModulePatchHook(patches=module_patches))

        # (3) Trainable parameter hook is prepended next.
        freeze_patterns = tuple(normalized.get("freeze_patterns") or ())
        trainable_patterns = tuple(normalized.get("trainable_patterns") or ())
        needs_trainable_filter = (
            peft_config is not None
            or bool(module_patches)
            or bool(freeze_patterns)
            or bool(trainable_patterns)
        )
        if needs_trainable_filter:
            generated.append(
                TrainableParameterHook(
                    freeze_patterns=freeze_patterns,
                    trainable_patterns=trainable_patterns,
                    freeze_mode=normalized.get("freeze_mode", "requires_grad"),
                )
            )
            generated.append(FineTuningSummaryHook())

        if generated:
            normalized["hooks"] = [*generated, *list(normalized.get("hooks") or [])]
        return normalized

    @staticmethod
    def _normalize_parameter_registration(
        names: Sequence[str],
        *,
        source: str,
    ) -> tuple[str, frozenset[str]]:
        """Validate and normalize a parameter registration set."""
        if not isinstance(source, str) or not source:
            raise ValueError(
                "parameter registration source must be a non-empty string."
            )
        if not isinstance(names, Sequence) or isinstance(names, (str, bytes)):
            raise TypeError("parameter names must be a non-string sequence of strings.")
        normalized_names = frozenset(names)
        if not all(isinstance(name, str) for name in normalized_names):
            raise TypeError("parameter names must contain only strings.")
        return source, normalized_names

    def register_trainable_parameter_names(
        self,
        names: Sequence[str],
        *,
        source: str,
    ) -> None:
        """Register trainable parameter names for a given source.

        Registration-time hooks use this method to add parameter names to the
        final trainable allow-list computed by :class:`TrainableParameterHook`.

        Parameters
        ----------
        names : Sequence[str]
            Fully-qualified parameter names, for example
            ``"main.model.projection.weight"``.
        source : str
            Identifier for the hook or feature that owns this registration,
            for example ``"module_patch"``.

        Returns
        -------
        None
            This method mutates ``self._registered_trainable_parameter_names``.

        Raises
        ------
        TypeError
            If ``names`` is not a non-string sequence, or if any entry in
            ``names`` is not a string.
        ValueError
            If ``source`` is not a non-empty string.
        """
        source, normalized_names = self._normalize_parameter_registration(
            names,
            source=source,
        )
        existing_names = self._registered_trainable_parameter_names.get(
            source,
            frozenset(),
        )
        self._registered_trainable_parameter_names[source] = (
            existing_names | normalized_names
        )

    def register_managed_parameter_names(
        self,
        names: Sequence[str],
        *,
        source: str,
    ) -> None:
        """Register managed parameter names for a given source so that a new
        source can choose to not override them.

        Parameters
        ----------
        names : Sequence[str]
            Fully-qualified parameter names, for example
            ``"main.model.projection.base_layer.weight"``.
        source : str
            Identifier for the hook or feature that owns this registration,
            for example ``"module_patch"``.

        Returns
        -------
        None
            This method mutates ``self._registered_managed_parameter_names``.

        Raises
        ------
        TypeError
            If ``names`` is not a non-string sequence, or if any entry in
            ``names`` is not a string.
        ValueError
            If ``source`` is not a non-empty string.
        """
        source, normalized_names = self._normalize_parameter_registration(
            names,
            source=source,
        )
        existing_names = self._registered_managed_parameter_names.get(
            source,
            frozenset(),
        )
        self._registered_managed_parameter_names[source] = (
            existing_names | normalized_names
        )

    def get_registered_trainable_parameter_names(self) -> Mapping[str, frozenset[str]]:
        """Return registered trainable parameter names grouped by source.

        Returns
        -------
        Mapping[str, frozenset[str]]
            A shallow copy of the mapping from source identifier to immutable registered
            trainable parameter names.
        """
        return dict(self._registered_trainable_parameter_names)

    def get_registered_managed_parameter_names(self) -> Mapping[str, frozenset[str]]:
        """Return registered managed parameter names grouped by source.

        Returns
        -------
        Mapping[str, frozenset[str]]
            A shallow copy of the mapping from source identifier to immutable registered
            managed parameter names.
        """
        return dict(self._registered_managed_parameter_names)

    def get_flattened_registered_trainable_parameter_names(self) -> frozenset[str]:
        """Return registered trainable parameter names across all sources.

        Returns
        -------
        frozenset[str]
            Immutable union of all registered trainable parameter names.
        """
        return frozenset().union(*self._registered_trainable_parameter_names.values())

    def get_flattened_registered_managed_parameter_names(self) -> frozenset[str]:
        """Return registered managed parameter names across all sources.

        Returns
        -------
        frozenset[str]
            Immutable union of all registered managed parameter names.
        """
        return frozenset().union(*self._registered_managed_parameter_names.values())

    def register_base_fingerprints(
        self,
        fingerprints: Mapping[str, str],
    ) -> None:
        """Register base-model fingerprints taken before architecture changes.

        A fingerprint hashes the model architecture, so it must be taken while
        the base model is still pristine, before fine-tuning modifies it by
        injecting adapters or patching modules. Currently used only by the
        parameter-efficient fine-tuning (PEFT) workflow, which records the
        fingerprints in the strategy spec so a checkpoint can be checked
        against the base architecture it was trained on.

        Parameters
        ----------
        fingerprints : Mapping[str, str]
            Architecture hash of each pristine base model, keyed by model name.

        Returns
        -------
        None
            This method replaces ``self._base_fingerprints`` with a copy of
            ``fingerprints``.
        """
        self._base_fingerprints = dict(fingerprints)

    def register_trainable_parameter_summary(
        self,
        summary: Mapping[str, Mapping[str, int]],
    ) -> None:
        """Register final trainable-parameter counts.

        Parameters
        ----------
        summary : Mapping[str, Mapping[str, int]]
            ``{"tensor_count": ..., "parameter_count": ...}`` counts keyed by
            registration source.

        Returns
        -------
        None
            This method replaces ``self._trainable_parameter_summary`` with a
            copy of ``summary``.
        """
        self._trainable_parameter_summary = {
            source: dict(counts) for source, counts in summary.items()
        }

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
        if self.peft_config is not None:
            spec["peft_config"] = self.peft_config.to_spec_dict()
            if self._peft_details:
                spec["peft_details"] = dict(self._peft_details)
            spec["compute_base_fingerprints"] = self.compute_base_fingerprints
            if self._base_fingerprints:
                spec["base_model_fingerprints"] = dict(self._base_fingerprints)
        if self._trainable_parameter_summary:
            spec["trainable_parameter_summary"] = self._trainable_parameter_summary
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

        # Parse PEFT-related configuration and validate base model fingerprints if needed.
        peft_config = None
        raw_peft_config = spec.get("peft_config")
        if raw_peft_config is not None:
            if not isinstance(raw_peft_config, Mapping):
                raise ValueError("from_spec_dict: peft_config must be a mapping.")
            peft_config = PeftConfig.from_spec_dict(raw_peft_config)
        compute_base_fingerprints = spec.get("compute_base_fingerprints", True)
        if peft_config is not None and compute_base_fingerprints:
            if "base_model_fingerprints" not in spec:
                raise ValueError(
                    "from_spec_dict: PEFT specs with compute_base_fingerprints=True "
                    "must include base_model_fingerprints."
                )
            validate_base_fingerprints(model_input, spec["base_model_fingerprints"])

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
            peft_config=peft_config,
            module_patches=module_patches,
            freeze_patterns=tuple(spec.get("freeze_patterns", ())),
            trainable_patterns=tuple(spec.get("trainable_patterns", ())),
            freeze_mode=spec.get("freeze_mode", "requires_grad"),
            compute_base_fingerprints=compute_base_fingerprints,
        )
