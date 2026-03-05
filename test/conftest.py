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

import pytest
import torch


@pytest.fixture(params=["cpu", "cuda"])
def device(request) -> str:
    """Return either CPU or GPU device; skips GPU if torch.cuda is unavailable."""
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("No CUDA device available.")
    return request.param


@pytest.fixture(params=["cuda"])
def gpu_device(request) -> str:
    """Used to skip GPU specific tests if device is not available."""
    if not torch.cuda.is_available():
        pytest.skip("No CUDA device available for GPU test.")
    return request.param
