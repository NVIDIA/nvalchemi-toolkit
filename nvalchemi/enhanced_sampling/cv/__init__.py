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
"""Collective-variable functions for enhanced sampling.

CVs are plain callables — no class hierarchy, no registration.  Any
differentiable function ``cv(batch: Batch) -> Tensor[B, D]`` satisfies
the CV interface.

Available CVs: :func:`pair_distance`, with :func:`pair_displacement` for
methods that work with the CV *gradient* rather than its value.
:func:`periodic_difference` is a helper for comparing CV values that live
on a circle.
"""

from nvalchemi.enhanced_sampling.cv._periodic import periodic_difference
from nvalchemi.enhanced_sampling.cv.pair_distance import (
    pair_displacement,
    pair_distance,
)

__all__ = ["pair_displacement", "pair_distance", "periodic_difference"]
