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

"""Unit tests for ``nvalchemi.distributed._core.op_transforms`` — the
argument / output transform markers carried on an ``OpAdapter``.
"""

from __future__ import annotations

import pytest

from nvalchemi.distributed._core.op_transforms import (
    AllReduceSum,
    GatherInputs,
    GatherInputsFull,
    ScatterOutputs,
    SliceOutputsOwned,
    SliceOwned,
)


class TestTransforms:
    def test_arg_transform_isinstance_dispatch(self):
        for t in (GatherInputs(), GatherInputsFull(), SliceOwned()):
            assert isinstance(t, (GatherInputs, GatherInputsFull, SliceOwned))

    def test_output_transform_isinstance_dispatch(self):
        for t in (ScatterOutputs(), AllReduceSum(), SliceOutputsOwned()):
            assert isinstance(t, (ScatterOutputs, AllReduceSum, SliceOutputsOwned))

    def test_frozen(self):
        # All transforms are frozen dataclasses; mutation should raise.
        with pytest.raises(Exception):
            GatherInputs().some_attr = 1  # type: ignore[attr-defined]

    def test_equal_by_class(self):
        # Frozen dataclasses with no fields compare equal; useful for
        # testing transform-table equality.
        assert GatherInputs() == GatherInputs()
        assert AllReduceSum() == AllReduceSum()
        assert GatherInputs() != GatherInputsFull()
