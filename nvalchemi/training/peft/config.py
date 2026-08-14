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
"""Public PEFT configuration objects."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class PeftConfig(BaseModel):
    """Base class for ALCHEMI parameter-efficient fine-tuning configs.

    Parameters
    ----------
    peft_method : str
        Frozen PEFT method name.
    """

    peft_method: str = Field(frozen=True)

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    def to_spec_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation of this configuration.

        Returns
        -------
        dict[str, Any]
            JSON-safe PEFT configuration data suitable for serialization.
        """
        from nvalchemi.training.peft.registry import get_peft_registration_by_config

        registration = get_peft_registration_by_config(self)
        config_data = self.model_dump(mode="python")
        if self.peft_method != registration.peft_method:
            raise ValueError(
                f"{type(self).__name__}.peft_method must match its registered "
                f"method {registration.peft_method!r}; got {self.peft_method!r}."
            )
        return config_data

    @classmethod
    def from_spec_dict(
        cls,
        spec: Mapping[str, Any],
        *,
        import_path_validator: Callable[[str, str], None] | None = None,
    ) -> PeftConfig:
        """Rebuild a PEFT config from its JSON-safe representation.

        Parameters
        ----------
        spec : Mapping[str, Any]
            Configuration containing a ``peft_method`` discriminator and
            method-specific fields.
        import_path_validator : Callable[[str, str], None] | None, optional
            Optional validator called before importing referenced classes.

        Returns
        -------
        PeftConfig
            The concrete registered PEFT configuration.
        """
        method = spec.get("peft_method")
        from nvalchemi.training.peft.registry import (
            available_peft_methods,
            get_peft_registration_by_method,
        )

        if not isinstance(method, str) or not method:
            raise ValueError(
                "PEFT config specs must contain a non-empty string peft_method; "
                f"available methods: {list(available_peft_methods())}."
            )
        registration = get_peft_registration_by_method(method)
        if cls is not PeftConfig and not issubclass(registration.config_cls, cls):
            raise ValueError(
                f"PEFT method {method!r} resolves to "
                f"{registration.config_cls.__name__}, not {cls.__name__}."
            )
        return registration.config_cls.model_validate(
            spec,
            context={"import_path_validator": import_path_validator},
        )


# ---------------------------------------------------------------------------
# Strategy registration
# ---------------------------------------------------------------------------


def build_peft_setup_hooks(
    config: PeftConfig,
    strategy_data: Mapping[str, Any],
) -> list[Any]:
    """Build registration-time hooks for a PEFT config.

    Parameters
    ----------
    config : PeftConfig
        The PEFT config that determines which hooks should be built.
    strategy_data : Mapping[str, Any]
        Strategy-specific setup data passed through to the PEFT method.

    Returns
    -------
    list[Any]
        Registration-time hooks required by the PEFT method.
    """
    from nvalchemi.training.peft.registry import get_peft_registration_by_config

    registration = get_peft_registration_by_config(config)
    return registration.build_peft_setup_hooks(config, strategy_data)
