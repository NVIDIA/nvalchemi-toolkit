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

from __future__ import annotations

from typing import TYPE_CHECKING

# Use lazy imports to avoid ModuleNotFoundError for missing model implementations
# (aimnet2, mace) in this worktree. The imports are deferred to attribute access.

if TYPE_CHECKING:
    from nvalchemi.models.additive import AdditiveModelWrapper
    from nvalchemi.models.aimnet2 import AIMNet2, AIMNet2Wrapper
    from nvalchemi.models.demo import DemoModelWrapper
    from nvalchemi.models.dftd3 import DFTD3ModelWrapper
    from nvalchemi.models.ewald import EwaldModelWrapper
    from nvalchemi.models.lj import LennardJonesModelWrapper
    from nvalchemi.models.mace import MACEWrapper
    from nvalchemi.models.pme import PMEModelWrapper

__all__ = [
    "AdditiveModelWrapper",
    "DemoModelWrapper",
    "DFTD3ModelWrapper",
    "EwaldModelWrapper",
    "LennardJonesModelWrapper",
    "PMEModelWrapper",
    "AIMNet2",
    "AIMNet2Wrapper",
    "MACEWrapper",
]


def __getattr__(name: str):
    """Lazy import to handle missing optional model implementations."""
    if name in ("AIMNet2", "AIMNet2Wrapper"):
        from nvalchemi.models.aimnet2 import AIMNet2, AIMNet2Wrapper

        return {"AIMNet2": AIMNet2, "AIMNet2Wrapper": AIMNet2Wrapper}[name]
    elif name == "AdditiveModelWrapper":
        from nvalchemi.models.additive import AdditiveModelWrapper

        return AdditiveModelWrapper
    elif name == "DemoModelWrapper":
        from nvalchemi.models.demo import DemoModelWrapper

        return DemoModelWrapper
    elif name == "DFTD3ModelWrapper":
        from nvalchemi.models.dftd3 import DFTD3ModelWrapper

        return DFTD3ModelWrapper
    elif name == "EwaldModelWrapper":
        from nvalchemi.models.ewald import EwaldModelWrapper

        return EwaldModelWrapper
    elif name == "LennardJonesModelWrapper":
        from nvalchemi.models.lj import LennardJonesModelWrapper

        return LennardJonesModelWrapper
    elif name == "MACEWrapper":
        from nvalchemi.models.mace import MACEWrapper

        return MACEWrapper
    elif name == "PMEModelWrapper":
        from nvalchemi.models.pme import PMEModelWrapper

        return PMEModelWrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
