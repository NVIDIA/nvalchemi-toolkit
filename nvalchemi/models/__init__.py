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
    from nvalchemi.models.aimnet2 import AIMNet2, AIMNet2Wrapper
    from nvalchemi.models.demo import DemoModelWrapper
    from nvalchemi.models.mace import ScaleShiftMACE, ScaleShiftMACEWrapper

__all__ = [
    "DemoModelWrapper",
    "AIMNet2",
    "AIMNet2Wrapper",
    "ScaleShiftMACE",
    "ScaleShiftMACEWrapper",
]


def __getattr__(name: str):
    """Lazy import to handle missing optional model implementations."""
    if name in ("AIMNet2", "AIMNet2Wrapper"):
        from nvalchemi.models.aimnet2 import AIMNet2, AIMNet2Wrapper

        return {"AIMNet2": AIMNet2, "AIMNet2Wrapper": AIMNet2Wrapper}[name]
    elif name == "DemoModelWrapper":
        from nvalchemi.models.demo import DemoModelWrapper

        return DemoModelWrapper
    elif name in ("ScaleShiftMACE", "ScaleShiftMACEWrapper"):
        from nvalchemi.models.mace import ScaleShiftMACE, ScaleShiftMACEWrapper

        return {
            "ScaleShiftMACE": ScaleShiftMACE,
            "ScaleShiftMACEWrapper": ScaleShiftMACEWrapper,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
