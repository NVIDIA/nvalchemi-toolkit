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
"""Model-side generative API.

This subpackage holds the model-specific generative surface — the
non-energy counterpart to :mod:`nvalchemi.models.base`. The mixin owns
only the raw model output (predicted flow/velocity) from one forward
call, plus the model↔:class:`~nvalchemi.data.Batch` translation methods.
It owns no scheduler, sampler, or guidance; those compose on the
:class:`~nvalchemi.gen.generator.AtomGenerator` via a
:class:`~nvalchemi.gen.generator.GeneratingFunction`.
"""

from __future__ import annotations

from nvalchemi.gen.enums import GenerativeIntent, Modality
from nvalchemi.models.gen.base import (
    ArtifactT,
    GenerativeModelConfig,
    GenerativeModelMixin,
)
from nvalchemi.models.gen.demo import (
    DemoDiffusionModel,
    DemoGANModel,
    demo_nonparametric_generation,
)

__all__ = [
    "ArtifactT",
    "DemoDiffusionModel",
    "DemoGANModel",
    "GenerativeIntent",
    "GenerativeModelConfig",
    "GenerativeModelMixin",
    "Modality",
    "demo_nonparametric_generation",
]
