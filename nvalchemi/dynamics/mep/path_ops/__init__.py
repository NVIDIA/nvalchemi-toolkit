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

"""Low-level tensor operations shared by reaction-path methods.

Planned toolkit-ops locations
-----------------------------
``torch_paths.py`` PyTorch validation and custom-op boundary
    ``nvalchemiops/torch/dynamics/paths.py``
``paths.py`` Warp kernels and overload registration
    ``nvalchemiops/dynamics/paths.py``
"""

from __future__ import annotations
