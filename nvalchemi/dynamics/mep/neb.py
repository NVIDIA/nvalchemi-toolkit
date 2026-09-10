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

"""Nudged elastic band strategy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_serializer,
    field_validator,
    model_validator,
)

from nvalchemi._serialization import SerializableClass
from nvalchemi.dynamics.base import ConvergenceHook, FusedStage
from nvalchemi.dynamics.hooks import FreezeAtomsHook, LoggingHook
from nvalchemi.dynamics.optimizers.fire2 import FIRE2
from nvalchemi.dynamics.paths.hooks import (
    PathDiagnosticsHook,
    PathEnergyStatsHook,
)
from nvalchemi.dynamics.paths.neb.configs import (
    ConstantSpringConfig,
    NEBMethod,
    SpringConfig,
)
from nvalchemi.dynamics.paths.neb.hooks import (
    ClimbingImageSelectionHook,
    NEBForceHook,
)
from nvalchemi.dynamics.strategy import (
    DynamicsStrategy,
    PositiveInt,
    _build_spec_component,
    _component_spec_dict,
)
from nvalchemi.hooks import Hook

if TYPE_CHECKING:
    from nvalchemi.models.base import BaseModelMixin

__all__ = ["ClimbingImageConfig", "NEB"]


class ClimbingImageConfig(BaseModel):
    """Configure climbing-image NEB activation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Literal["after_regular", "immediate"] = Field(
        default="after_regular",
        description=(
            "Activate climbing forces after regular NEB convergence or from "
            "the initial evaluation."
        ),
    )
    selection: Literal["fixed", "dynamic"] = Field(
        default="fixed",
        description=(
            "Keep the initially selected climbing image or reselect the "
            "highest-energy image after every evaluation."
        ),
    )
    regular_fmax: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Force threshold for entering the climbing stage; None uses the "
            "final NEB fmax. Only valid with mode='after_regular'."
        ),
    )
    max_regular_steps: PositiveInt | None = Field(
        default=None,
        description=(
            "Per-path step budget for regular NEB. When reached, the path enters "
            "the climbing-image stage even if the regular-stage convergence "
            "criterion has not been satisfied. None waits for regular-stage "
            "convergence, subject to the overall NEB.n_steps limit. "
            "Only valid with mode='after_regular'. "
        ),
    )
    max_climbing_steps: PositiveInt | None = Field(
        default=None,
        description="Per-path step budget for climbing-image optimization.",
    )

    @field_validator("regular_fmax", mode="before")
    @classmethod
    def _validate_regular_fmax(cls, value: Any) -> float | None:
        """Validate and normalize the regular-stage convergence threshold."""
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("regular_fmax must be a number")
        if value <= 0:
            raise ValueError("regular_fmax must be positive")
        return float(value)

    @model_validator(mode="after")
    def _validate_regular_stage_options(self) -> Self:
        """Restrict regular-stage options to staged climbing strategies."""
        if self.mode != "after_regular" and self.regular_fmax is not None:
            raise ValueError("regular_fmax is only valid with mode='after_regular'")
        if self.mode != "after_regular" and self.max_regular_steps is not None:
            raise ValueError(
                "max_regular_steps is only valid with mode='after_regular'"
            )
        return self


_NEB_MANAGED_OPTIMIZER_KWARGS = {
    "by_group",
    "convergence_hook",
    "hooks",
    "model",
    "n_steps",
}


class NEB(DynamicsStrategy):
    """Run regular or climbing-image nudged elastic band optimization.

    Every graph in the input batch is one path image, and its group layout
    identifies complete paths. The strategy builds a group-aware
    :class:`~nvalchemi.dynamics.FusedStage` containing one or two
    optimizer stages.

    Attributes
    ----------
    _convergence_hook : ConvergenceHook or None
        Optional override for the final or only optimizer stage. When it is
        ``None``, :meth:`build_engine` constructs the force criterion from
        ``fmax`` without storing it on the strategy. Overrides must use the
        standard hook type with ``frequency=1``.
    _regular_convergence_hook : ConvergenceHook or None
        Optional override for the preliminary stage of an ``after_regular`` run.
        When it is ``None``, :meth:`build_engine` constructs the force criterion
        from ``climbing.regular_fmax`` or ``fmax`` without storing it on the
        strategy. It has the same type and frequency constraints and is otherwise
        ignored.
    """

    model_config = ConfigDict(validate_assignment=True)

    spring: float | SpringConfig = Field(
        default=0.1,
        description="Spring configuration used by the shared NEB force hook.",
    )
    method: str | NEBMethod = Field(
        default="improved_tangent",
        description="NEB tangent and spring-force formulation.",
    )
    climbing: ClimbingImageConfig | None = Field(
        default=None,
        description="Climbing-image configuration; None runs regular NEB only.",
    )
    optimizer: SerializableClass = Field(
        default=FIRE2,
        description=(
            "Optimizer class used by internal stages. Only FIRE2 is currently "
            "supported."
        ),
    )
    optimizer_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Keyword arguments forwarded to each internal optimizer stage. "
            "When using FIRE2, defaults include dt = 0.01."
        ),
    )
    fmax: float = Field(
        default=0.05,
        gt=0,
        description="Force threshold for convergence of the final or only stage.",
    )
    endpoint_mode: Literal["fixed", "relaxed"] = Field(
        default="fixed",
        description="Whether path endpoint coordinates remain fixed or are relaxed.",
    )
    fixed_atom_indices: dict[int, tuple[int, ...]] | None = Field(
        default=None,
        description=(
            "Mapping from path indices to atom indices held fixed in every image "
            "of that path."
        ),
    )
    diagnostics_log_path: Path | None = Field(
        default=None,
        description=(
            "CSV output path for per-path diagnostics. None disables both path "
            "diagnostics and their logging hook."
        ),
    )
    diagnostics_frequency: PositiveInt = Field(
        default=1,
        description="Step frequency for computing and logging per-path diagnostics.",
    )
    compile: bool = Field(
        default=False,
        strict=True,
        description="Compile the fused NEB step with torch.compile.",
    )
    compile_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments forwarded to torch.compile.",
    )
    neighbor_hooks: list[Hook] | None = Field(
        default=None,
        exclude=True,
        description=(
            "Neighbor-list hooks for the fused NEB engine. None uses hooks "
            "generated by model.make_neighbor_hooks()."
        ),
    )
    _convergence_hook: ConvergenceHook | None = PrivateAttr(default=None)
    _regular_convergence_hook: ConvergenceHook | None = PrivateAttr(default=None)

    @field_validator("fmax", mode="before")
    @classmethod
    def _validate_fmax(cls, value: Any) -> float:
        """Validate and normalize the convergence threshold."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("fmax must be a number")
        if value <= 0:
            raise ValueError("fmax must be positive")
        return float(value)

    @field_validator("compile", mode="before")
    @classmethod
    def _validate_boolean_options(cls, value: Any) -> bool:
        """Reject integer coercion for the compile option."""
        if not isinstance(value, bool):
            raise TypeError("compile must be boolean")
        return value

    @field_validator("optimizer_kwargs", mode="before")
    @classmethod
    def _normalize_optimizer_kwargs(cls, value: Any) -> dict[str, Any]:
        """Normalize optimizer keyword arguments."""
        if value is None:
            normalized: dict[str, Any] = {}
        elif isinstance(value, dict):
            normalized = dict(value)
        else:
            raise TypeError("optimizer_kwargs must be a dictionary or None")
        reserved = _NEB_MANAGED_OPTIMIZER_KWARGS & normalized.keys()
        if reserved:
            names = ", ".join(sorted(reserved))
            raise ValueError(
                f"optimizer_kwargs cannot override NEB-managed keys: {names}"
            )
        return normalized

    @field_validator("neighbor_hooks", mode="before")
    @classmethod
    def _normalize_neighbor_hooks(cls, value: Any) -> list[Hook] | None:
        """Validate and copy explicit neighbor hooks, preserving ``None`` to use
        model-generated hooks."""
        if value is None:
            return None
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError("neighbor_hooks must be a sequence of Hook objects or None")
        hooks = list(value)
        if any(not isinstance(hook, Hook) for hook in hooks):
            raise TypeError("neighbor_hooks must contain only Hook objects")
        return hooks

    @staticmethod
    def _validate_convergence_hook(value: Any) -> None:
        """Validate one convergence override at engine build time."""
        if not isinstance(value, ConvergenceHook):
            raise TypeError(
                "NEB convergence overrides must be ConvergenceHook instances"
            )
        if type(value) is not ConvergenceHook:
            raise TypeError("NEB convergence overrides do not support hook subclasses")
        if value.frequency != 1:
            raise ValueError("NEB convergence overrides must use frequency=1")
        if value.source_status is not None or value.target_status is not None:
            raise ValueError(
                "NEB convergence hooks cannot set source_status or target_status; "
                "NEB manages stage migration"
            )
        if not value.by_group:
            raise ValueError("NEB convergence hooks must use by_group=True")

    @field_validator("compile_kwargs", mode="before")
    @classmethod
    def _normalize_compile_kwargs(cls, value: Any) -> dict[str, Any]:
        """Own compile kwargs while accepting ``None`` as the default."""
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise TypeError("compile_kwargs must be a dictionary or None")
        return dict(value)

    @field_validator("fixed_atom_indices", mode="before")
    @classmethod
    def _validate_fixed_atom_indices(cls, value: Any) -> Any:
        """Normalize the per-path fixed-atom mapping."""
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise TypeError(
                "fixed_atom_indices must map path indices to atom-index sequences"
            )
        try:
            normalized = {
                path_index: tuple(atom_indices)
                for path_index, atom_indices in value.items()
            }
        except TypeError as exc:
            raise TypeError(
                "fixed_atom_indices values must be sequences of integers"
            ) from exc
        if any(
            isinstance(path_index, bool) or not isinstance(path_index, int)
            for path_index in normalized
        ):
            raise TypeError("fixed_atom_indices keys must be integer path indices")
        if any(
            isinstance(atom_index, bool) or not isinstance(atom_index, int)
            for atom_indices in normalized.values()
            for atom_index in atom_indices
        ):
            raise TypeError("fixed_atom_indices values must contain only integers")
        return normalized

    @field_validator("spring", mode="before")
    @classmethod
    def _validate_spring(cls, value: Any) -> Any:
        """Validate spring policies and restore built-in constant recipes."""
        if isinstance(value, dict) and value.get("type") == "constant":
            if set(value) != {"type", "value"}:
                raise ValueError("constant spring spec must contain type and value")
            value = ConstantSpringConfig(value["value"])
        if isinstance(value, bool):
            raise TypeError("spring must be a number or implement SpringConfig")
        if isinstance(value, (int, float)):
            ConstantSpringConfig(value)
            return float(value)
        if not isinstance(value, SpringConfig):
            raise TypeError("spring must be a number or implement SpringConfig")
        return value

    @field_serializer("spring", when_used="json")
    def _serialize_spring(self, spring: float | SpringConfig) -> float | dict[str, Any]:
        """Serialize numeric and built-in constant spring configurations."""
        if isinstance(spring, (int, float)):
            return float(spring)
        if isinstance(spring, ConstantSpringConfig):
            return {"type": "constant", "value": spring.value}
        raise ValueError(
            f"Spring policy {type(spring).__name__} is runtime-only and cannot be "
            "serialized by NEB.to_spec_dict()"
        )

    @field_serializer("method", when_used="json")
    def _serialize_method(self, method: str | NEBMethod) -> str:
        """Serialize NEB methods through their stable registry names."""
        if isinstance(method, str):
            return method
        if method.name is None:
            raise ValueError(
                "Unnamed NEBMethod objects are runtime-only and cannot be "
                "serialized by NEB.to_spec_dict()"
            )
        return method.name

    @model_validator(mode="after")
    def _validate_configuration(self) -> NEB:
        """Validate cross-field constraints and supported runtime types."""
        if self.optimizer is FIRE2:
            self.optimizer_kwargs.setdefault("dt", 0.01)
        else:
            raise NotImplementedError(
                f"Unsupported NEB optimizer: {self.optimizer.__qualname__}"
            )
        return self

    def _build_convergence_hook(
        self,
        *,
        fmax: float,
        regular_stage: bool,
    ) -> ConvergenceHook:
        """Resolve the group-aware convergence hook for one optimizer stage."""
        template = (
            self._regular_convergence_hook if regular_stage else self._convergence_hook
        )
        if template is None:
            return ConvergenceHook.from_fmax(threshold=fmax, by_group=True)
        self._validate_convergence_hook(template)
        return deepcopy(template)

    def _build_path_hooks(self, *, climbing_status: int | None) -> list[Hook]:
        """Build the shared ordered path hook stack."""
        energy_stats = PathEnergyStatsHook()
        hooks: list[Hook] = [energy_stats]
        if climbing_status is not None:
            config = self.climbing
            if config is None:
                raise RuntimeError("climbing stage requires climbing configuration")
            hooks.append(
                ClimbingImageSelectionHook(
                    energy_stats_hook=energy_stats,
                    selection=config.selection,
                    # Fused hooks receive the union mask, so restrict selection
                    # to the climbing-stage status.
                    status_code=climbing_status,
                )
            )
        hooks.append(
            NEBForceHook(
                energy_stats_hook=energy_stats,
                spring=self.spring,
                method=self.method,
                endpoint_mode=self.endpoint_mode,
                fixed_atom_indices=self.fixed_atom_indices,
            )
        )
        if self.endpoint_mode == "fixed" or self.fixed_atom_indices:
            hooks.append(
                FreezeAtomsHook(
                    mask_key="neb_fixed_node_mask",
                    zero_velocities=("velocities" in self.optimizer.__provides_keys__),
                )
            )
        if self.diagnostics_log_path is not None:
            diagnostics_hook = PathDiagnosticsHook(
                energy_stats_hook=energy_stats,
                frequency=self.diagnostics_frequency,
            )
            hooks.extend(
                [
                    diagnostics_hook,
                    LoggingHook(
                        backend="csv",
                        frequency=self.diagnostics_frequency,
                        log_path=self.diagnostics_log_path,
                        custom_scalars={
                            "fmax": (
                                lambda _ctx: diagnostics_hook.get_diagnostics().fmax
                            ),
                            "energy_barrier": (
                                lambda _ctx: (
                                    diagnostics_hook.get_diagnostics().energy_barrier
                                )
                            ),
                            "highest_interior_image_idx": (
                                lambda _ctx: (
                                    diagnostics_hook.get_diagnostics().highest_interior_image_idx
                                )
                            ),
                            "path_length": (
                                lambda _ctx: (
                                    diagnostics_hook.get_diagnostics().path_length
                                )
                            ),
                        },
                        by_group=True,
                    ),
                ]
            )
        return hooks

    def build_engine(self) -> FusedStage:
        """Construct a fresh fused optimizer.

        Returns
        -------
        FusedStage
            Group-aware one- or two-stage optimizer strategy.
        """
        if self.climbing is not None and self.climbing.mode == "after_regular":
            climbing_status = 1
            stages = [
                (
                    0,
                    self.optimizer(
                        model=self.model,
                        n_steps=self.climbing.max_regular_steps,
                        by_group=True,
                        convergence_hook=self._build_convergence_hook(
                            fmax=(
                                self.fmax
                                if self.climbing.regular_fmax is None
                                else self.climbing.regular_fmax
                            ),
                            regular_stage=True,
                        ),
                        **self.optimizer_kwargs,
                    ),
                ),
                (
                    1,
                    self.optimizer(
                        model=self.model,
                        n_steps=self.climbing.max_climbing_steps,
                        by_group=True,
                        convergence_hook=self._build_convergence_hook(
                            fmax=self.fmax,
                            regular_stage=False,
                        ),
                        **self.optimizer_kwargs,
                    ),
                ),
            ]
            reprime_on_entry = {1}
        else:
            climbing_status = 0 if self.climbing is not None else None
            stages = [
                (
                    0,
                    self.optimizer(
                        model=self.model,
                        n_steps=(
                            None
                            if self.climbing is None
                            else self.climbing.max_climbing_steps
                        ),
                        by_group=True,
                        convergence_hook=self._build_convergence_hook(
                            fmax=self.fmax,
                            regular_stage=False,
                        ),
                        **self.optimizer_kwargs,
                    ),
                )
            ]
            reprime_on_entry = set()

        neighbor_hooks = (
            self.model.make_neighbor_hooks()
            if self.neighbor_hooks is None
            else self.neighbor_hooks
        )
        fused_hooks = [
            *neighbor_hooks,
            *self._build_path_hooks(climbing_status=climbing_status),
            *self.extra_hooks,
        ]
        return FusedStage(
            sub_stages=stages,
            n_steps=self.n_steps,
            by_group=True,
            hooks=fused_hooks,
            reprime_on_entry=reprime_on_entry,
            compile_step=self.compile,
            compile_kwargs=dict(self.compile_kwargs),
            device_type=stages[0][1].device_type,
        )

    def to_spec_dict(self) -> dict[str, Any]:
        """Serialize NEB configuration and convergence hooks."""
        spec = super().to_spec_dict()
        spec["neighbor_hook_specs"] = (
            None
            if self.neighbor_hooks is None
            else [
                _component_spec_dict(hook, label=f"neighbor hook at index {index}")
                for index, hook in enumerate(self.neighbor_hooks)
            ]
        )
        for field_name in ("convergence_hook", "regular_convergence_hook"):
            hook = getattr(self, f"_{field_name}")
            spec[f"{field_name}_spec"] = (
                None if hook is None else _component_spec_dict(hook, label=field_name)
            )
        return spec

    @classmethod
    def from_spec_dict(
        cls,
        spec: Mapping[str, Any],
        *,
        model: BaseModelMixin,
        extra_hooks: Sequence[Hook] | None = None,
    ) -> Self:
        """Rebuild NEB configuration, including convergence hooks."""
        if not isinstance(spec, Mapping):
            raise TypeError("spec must be a mapping")
        data = dict(spec)
        raw_neighbor_hook_specs = data.pop("neighbor_hook_specs", None)
        if raw_neighbor_hook_specs is not None:
            if not isinstance(raw_neighbor_hook_specs, list):
                raise ValueError(
                    "from_spec_dict: neighbor_hook_specs must be null or a list "
                    "of constructor specs."
                )
            data["neighbor_hooks"] = [
                _build_spec_component(
                    raw_hook_spec,
                    label=f"neighbor hook at index {index}",
                )
                for index, raw_hook_spec in enumerate(raw_neighbor_hook_specs)
            ]
        fixed_atom_indices = data.get("fixed_atom_indices")
        if isinstance(fixed_atom_indices, Mapping):
            data["fixed_atom_indices"] = {
                int(path_index)
                if isinstance(path_index, str) and path_index.lstrip("-").isdigit()
                else path_index: atom_indices
                for path_index, atom_indices in fixed_atom_indices.items()
            }
        overrides: dict[str, ConvergenceHook] = {}
        for field_name in ("convergence_hook", "regular_convergence_hook"):
            if field_name in data:
                raise ValueError(f"spec cannot contain runtime field: {field_name}")
            raw_spec = data.pop(f"{field_name}_spec", None)
            if raw_spec is None:
                continue
            hook = _build_spec_component(raw_spec, label=f"{field_name}_spec")
            if not isinstance(hook, ConvergenceHook):
                raise TypeError(
                    f"{field_name}_spec built {type(hook).__name__}, expected "
                    "ConvergenceHook"
                )
            overrides[field_name] = hook
        rebuilt = super().from_spec_dict(
            data,
            model=model,
            extra_hooks=extra_hooks,
        )
        if not isinstance(rebuilt, cls):
            raise RuntimeError(f"Expected {cls.__name__}, got {type(rebuilt).__name__}")
        for field_name, hook in overrides.items():
            setattr(rebuilt, f"_{field_name}", hook)
        return rebuilt
