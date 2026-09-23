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
"""Loss terms that compare a student against teacher signals.

The built-in terms in :mod:`nvalchemi.training.losses` already cover every
teacher signal that has a total-energy, force, or stress shape — point their
``target_key`` at the matching ``teacher_*`` field. This subpackage adds the
terms that have no supervised counterpart.
"""

from __future__ import annotations

from nvalchemi.training.distillation.losses.atomic_energy import (
    AtomicEnergyMatchingLoss,
)

__all__ = ["AtomicEnergyMatchingLoss"]
