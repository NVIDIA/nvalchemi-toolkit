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
"""Tests for the OptionalDependency guard mechanism."""

from __future__ import annotations

import pytest

from nvalchemi import OptionalDependency, OptionalDependencyError


def test_require_raises_for_missing_package():
    """Decorator raises OptionalDependencyError for a genuinely missing package."""
    fake = object.__new__(OptionalDependency)
    fake.import_name = "__nonexistent__"
    fake.install_target = "nvalchemi-toolkit[fake]"
    fake._available = None
    fake._import_error = None

    @fake.require
    def dummy():
        pass

    with pytest.raises(OptionalDependencyError):
        dummy()
