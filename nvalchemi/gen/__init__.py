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
"""Toolkit-level generative API surface.

This package provides the abstract
:class:`~nvalchemi.gen.generator.AtomGenerator` inference driver — a fixed
condition → generate pipeline with lifecycle hooks
(:class:`~nvalchemi.gen.stages.GenerationStage`,
:class:`~nvalchemi.hooks.GenerationContext`) — plus sequential
composition via :class:`~nvalchemi.gen.pipeline.GenerationPipeline`.
"""

from __future__ import annotations

from nvalchemi.gen.enums import GenerativeIntent, Modality
from nvalchemi.gen.generator import (
    AtomGenerator,
    GeneratingFunction,
    default_condition,
)
from nvalchemi.gen.pipeline import GenerationPipeline
from nvalchemi.gen.stages import GenerationStage
from nvalchemi.hooks import GenerationContext

__all__ = [
    "AtomGenerator",
    "GenerationContext",
    "GenerationPipeline",
    "GenerationStage",
    "GenerativeIntent",
    "GeneratingFunction",
    "Modality",
    "default_condition",
]
