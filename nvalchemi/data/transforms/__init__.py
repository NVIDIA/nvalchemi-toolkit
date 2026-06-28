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
"""User-supplied data transforms for :mod:`nvalchemi.data`.

This package hosts composition infrastructure for transforms that plug
into :class:`~nvalchemi.data.Dataset` (per-sample transforms) and
:class:`~nvalchemi.data.DataLoader` (per-batch transforms), and is the
intended home for future concrete transform submodules
"""

from __future__ import annotations

from nvalchemi.data.transforms.compose import Compose

__all__ = [
    "Compose",
]
