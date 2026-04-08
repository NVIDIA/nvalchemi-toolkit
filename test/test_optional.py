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

from nvalchemi._optional import OptionalDependency, OptionalDependencyError
from nvalchemi.data import AtomicData


@pytest.mark.parametrize(
    "dep,method",
    [
        (OptionalDependency.ASE, "from_atoms"),
        (OptionalDependency.PYMATGEN, "from_structure"),
    ],
)
def test_missing_optional_dep_raises(dep, method):
    """Guarded methods raise OptionalDependencyError when the dependency is missing."""
    original_available = dep._available
    original_error = dep._import_error
    try:
        dep._available = False
        dep._import_error = ImportError(f"No module named '{dep.import_name}'")
        with pytest.raises(OptionalDependencyError):
            getattr(AtomicData, method)(None)
    finally:
        dep._available = original_available
        dep._import_error = original_error
