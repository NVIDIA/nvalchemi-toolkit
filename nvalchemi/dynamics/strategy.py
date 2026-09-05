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
"""Declarative strategies that construct and run dynamics engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, TypeAlias

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator

from nvalchemi._serialization import _extract_init_kwargs_from_attrs
from nvalchemi.data import Batch
from nvalchemi.dynamics.base import BaseDynamics
from nvalchemi.hooks._protocol import Hook
from nvalchemi.models.base import BaseModelMixin
from nvalchemi.specs import BaseSpec, create_model_spec, create_model_spec_from_json

__all__ = ["DynamicsStrategy"]

# Strictly positive scalar integer
PositiveInt: TypeAlias = Annotated[int, Field(strict=True, gt=0)]


def _constructor_spec(component: Any) -> Any:
    """Build a reconstructible spec from a component's constructor state."""
    checkpoint_spec = getattr(component, "checkpoint_spec", None)
    if callable(checkpoint_spec):
        spec = checkpoint_spec()
        if spec is not None:
            if not isinstance(spec, BaseSpec):
                raise TypeError(
                    "checkpoint_spec() must return a BaseSpec or None; got "
                    f"{type(spec).__name__}."
                )
            return spec

    kwargs = _extract_init_kwargs_from_attrs(component)
    for name, value in list(kwargs.items()):
        if isinstance(value, torch.nn.Module):
            kwargs[name] = _constructor_spec(value)
    return create_model_spec(type(component), **kwargs)


def _component_spec_dict(component: Any, *, label: str) -> dict[str, Any]:
    """Serialize one reconstructible runtime component to a JSON dictionary."""
    try:
        return _constructor_spec(component).model_dump(mode="json")
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Cannot serialize {label} {type(component).__name__}: {exc}"
        ) from exc


def _build_spec_component(raw: Any, *, label: str) -> Any:
    """Build one runtime component from a serialized constructor spec."""
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"from_spec_dict: {label} must be a constructor-spec mapping; "
            f"got {type(raw).__name__}."
        )
    try:
        return create_model_spec_from_json(dict(raw)).build()
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"from_spec_dict: cannot rebuild {label}: {exc}") from exc


class DynamicsStrategy(BaseModel, ABC):
    """Base class for declarative, serializable dynamics strategies.

    A strategy stores the configuration needed to construct a fresh dynamics
    engine. :meth:`to_spec_dict` represents hooks as reconstructible constructor
    specs while the live model remains a runtime dependency. When restoring with
    :meth:`from_spec_dict`, ``model=`` supplies that dependency and
    ``extra_hooks=`` appends runtime hooks.

    Parameters
    ----------
    model : BaseModelMixin
        Potential model used by the dynamics engine.
    n_steps : int or None, optional
        Default number of dynamics steps. ``None`` delegates termination to
        the constructed engine.
    extra_hooks : sequence of Hook or None, optional
        Ordered runtime hooks added to the constructed engine.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_assignment=False,
        revalidate_instances="never",
    )

    model: BaseModelMixin = Field(
        exclude=True,
        description=(
            "Live potential model used by the dynamics engine; excluded from "
            "strategy serialization."
        ),
    )
    n_steps: PositiveInt | None = Field(
        default=None,
        description="Default dynamics step limit, or None for no fixed limit.",
    )
    extra_hooks: list[Hook] = Field(
        default_factory=list,
        exclude=True,
        description=(
            "Runtime hooks represented by ordered constructor specs during strategy "
            "serialization."
        ),
    )

    @field_validator("extra_hooks", mode="before")
    @classmethod
    def _normalize_extra_hooks(cls, value: Any) -> list[Hook]:
        """Convert an optional hook sequence to a new list."""
        if value is None:
            return []
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError("extra_hooks must be a sequence of Hook objects or None")
        return list(value)

    @abstractmethod
    def build_engine(self) -> BaseDynamics:
        """Construct a fresh dynamics engine.

        Returns
        -------
        BaseDynamics
            Newly constructed engine configured by this strategy.
        """

    def run(
        self,
        batch: Batch,
        n_steps: int | None = None,
    ) -> Batch:
        """Build a fresh engine and run it on ``batch``.

        Parameters
        ----------
        batch : Batch
            Atomic systems to update in place.
        n_steps : int or None, optional
            Per-run step-limit override. ``None`` uses :attr:`n_steps`.

        Returns
        -------
        Batch
            The input batch after dynamics updates.
        """
        engine = self.build_engine()
        result = engine.run(batch, n_steps=n_steps)
        if result is None:
            raise RuntimeError(
                f"{type(self).__name__} unexpectedly completed without a result batch"
            )
        return result

    def to_spec_dict(self) -> dict[str, Any]:
        """Serialize declarative strategy fields to a JSON-ready dictionary.

        Hooks are represented by constructor specs. The live model and mutable
        hook state are not included. Subclasses with arbitrary configuration
        objects must provide Pydantic serializers or override this method.

        Returns
        -------
        dict[str, Any]
            JSON-ready strategy configuration.
        """
        spec = self.model_dump(mode="json")
        spec["extra_hook_specs"] = [
            _component_spec_dict(hook, label=f"hook at index {index}")
            for index, hook in enumerate(self.extra_hooks)
        ]
        return spec

    @classmethod
    def from_spec_dict(
        cls,
        spec: Mapping[str, Any],
        *,
        model: BaseModelMixin,
        extra_hooks: Sequence[Hook] | None = None,
    ) -> DynamicsStrategy:
        """Rebuild a strategy from :meth:`to_spec_dict` output.

        Parameters
        ----------
        spec : mapping of str to Any
            JSON-decoded strategy configuration.
        model : BaseModelMixin
            Weighted potential model to attach to the restored strategy.
        extra_hooks : sequence of Hook or None, optional
            Runtime hooks appended after hooks rebuilt from serialized specs.

        Returns
        -------
        DynamicsStrategy
            Freshly validated strategy instance.
        """
        if not isinstance(spec, Mapping):
            raise TypeError("spec must be a mapping")
        data = dict(spec)
        runtime_fields = {"model", "extra_hooks"} & data.keys()
        if runtime_fields:
            names = ", ".join(sorted(runtime_fields))
            raise ValueError(f"spec cannot contain runtime-only fields: {names}")

        raw_hook_specs = data.pop("extra_hook_specs", [])
        if not isinstance(raw_hook_specs, list):
            raise ValueError(
                "from_spec_dict: extra_hook_specs must be a list of constructor specs."
            )
        hooks: list[Hook] = []
        for index, raw_hook_spec in enumerate(raw_hook_specs):
            hook = _build_spec_component(
                raw_hook_spec,
                label=f"extra_hook_specs[{index}]",
            )
            if not isinstance(hook, Hook):
                raise TypeError(
                    f"from_spec_dict: extra_hook_specs[{index}] built "
                    f"{type(hook).__name__}, expected Hook."
                )
            hooks.append(hook)

        data["model"] = model
        data["extra_hooks"] = [*hooks, *(extra_hooks or [])]
        return cls.model_validate(data)
