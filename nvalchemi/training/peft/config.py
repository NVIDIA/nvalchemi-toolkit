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

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class PeftConfig(BaseModel):
    """Base class for ALCHEMI parameter-efficient fine-tuning configs."""

    peft_method: str

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_spec_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation of this configuration.

        Returns
        -------
        dict[str, Any]
            JSON-safe PEFT configuration data suitable for serialization.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Metadata serialization
# ---------------------------------------------------------------------------


def peft_metadata_from_config(
    config: PeftConfig,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return JSON-safe PEFT metadata for a config object.

    Parameters
    ----------
    config : PeftConfig
        The PEFT config object to convert to metadata.
    details : Mapping[str, Any] | None, optional
        Additional details to include in the metadata.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the PEFT metadata.

    Notes
    -----
    PEFT metadata uses ``peft_method`` as the loading discriminator. Any
    recorded config class path is provenance metadata only and must not be
    dynamically imported during load.
    """
    from nvalchemi.training.peft.lora import LORA_PEFT_METHOD

    if config.peft_method == LORA_PEFT_METHOD:
        from nvalchemi.training.peft.lora import lora_metadata_from_config

        return lora_metadata_from_config(config, details)
    raise ValueError(f"Unsupported PEFT method {config.peft_method!r}.")


def peft_config_from_metadata(metadata: Mapping[str, Any]) -> PeftConfig:
    """Rebuild a PEFT config from serialized metadata.

    Parameters
    ----------
    metadata : Mapping[str, Any]
        Serialized PEFT metadata containing a ``peft_method`` discriminator and
        method-specific configuration payload.

    Returns
    -------
    PeftConfig
        The PEFT config reconstructed from the serialized metadata.
    """
    method = metadata.get("peft_method")
    from nvalchemi.training.peft.lora import LORA_PEFT_METHOD

    if method == LORA_PEFT_METHOD:
        from nvalchemi.training.peft.lora import lora_config_from_metadata

        return lora_config_from_metadata(metadata)
    raise ValueError(f"Unsupported PEFT method {method!r}.")


# ---------------------------------------------------------------------------
# Strategy registration
# ---------------------------------------------------------------------------


def peft_setup_hooks(
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
    from nvalchemi.training.peft.lora import LORA_PEFT_METHOD

    if config.peft_method == LORA_PEFT_METHOD:
        from nvalchemi.training.peft.lora import lora_setup_hooks

        return lora_setup_hooks(config, strategy_data)
    raise ValueError(f"Unsupported PEFT method {config.peft_method!r}.")
